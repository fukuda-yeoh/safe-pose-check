# main_live.py

import os
import threading
import json
import time

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv
import numpy as np
from threading import Thread

from components import (
    CameraThread,
    RecognitionThread,
    RecordWorker,
    EventDetector,
    EventState,
    DisplayFrame,
    play_buzzer_sound,
    play_ok_sound,
    play_detect_sound,
)
from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
)

# 設定ファイルのパスを定義
CONFIG_DIR = project_root() / "config"
CONFIG_FILE = CONFIG_DIR / "roi_config.json"

# カメラ行列と歪み係数
camera_matrix = np.array(
    [
        [635.8283543606941, 0.0, 253.67538054709416],
        [0.0, 636.5077974749544, 320.7767679600789],
        [0.0, 0.0, 1.0],
    ]
)

dist_coeffs = np.array(
    [
        -0.013855030664346785,
        0.036480883659782824,
        -0.004879021042728567,
        0.0019079058483680378,
        -0.21054703261551563,
    ],
    dtype=np.float32,
)

# カメラを開く
model_path = project_root() / "models" / "pose_landmarker_full.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

camera_thread = CameraThread(1, camera_matrix, dist_coeffs)
recognition_thread = RecognitionThread(model_path, camera_thread.queue)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")
record_count = 1
is_recording = False

# --- エリア定義ファイルの読み込み ---
roi_points = []
display_check_area_points = []

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        loaded_config = json.load(f)
        roi_points = loaded_config.get("roi_points", [])
        display_check_area_points = loaded_config.get("display_check_area_points", [])
    print(f"ROI configuration loaded from {CONFIG_FILE}")
else:
    print("ROI configuration file not found. Using default values.")

event_detector = EventDetector(
    landmark_labels,
    roi_points=roi_points,
    display_check_area_points=display_check_area_points,
)
detector_thread = Thread(
    target=event_detector.run_in_thread, args=(recognition_thread.out_queue,)
)

display_frame = DisplayFrame()
recognition_thread.start()
detector_thread.start()
camera_thread.start()

# ウィンドウの定義
cv.namedWindow("Monitor Window", cv.WINDOW_NORMAL)
cv.namedWindow("Manager Window", cv.WINDOW_NORMAL)

# --- 各種画像を読み込む ---
ok_img_path = project_root() / "OK.png"
ng_img_path = project_root() / "NG.png"
ankn_img_path = project_root() / "Area.png"
action_img_path = project_root() / "Action.png"
notice_img_path = project_root() / "Notice.png"

_ok_img = cv.imread(str(ok_img_path), cv.IMREAD_UNCHANGED)
_ng_img = cv.imread(str(ng_img_path), cv.IMREAD_UNCHANGED)
_ankn_img = cv.imread(str(ankn_img_path), cv.IMREAD_UNCHANGED)
_action_img = cv.imread(str(action_img_path), cv.IMREAD_UNCHANGED)
_notice_img = cv.imread(str(notice_img_path), cv.IMREAD_UNCHANGED)

_result_img_display_width = 300
_result_img_display_height = 300
_ankn_img_display_width = 400
_ankn_img_display_height = 400
_action_img_display_width = 400
_action_img_display_height = 400
_notice_img_display_width = 400
_notice_img_display_height = 400

# 判定結果表示エリア内でのシーケンス状態制御変数
_display_area_notice_start_time = 0.0
_is_notice_active = False
_is_replay_active = False
_has_saved_exit_status = False  # 進入時のステータス確定フラグ
_session_all_items_confirmed = False

_buzzer_played_for_current_ng = False
_ok_played_for_current_ok = False
_detect_played_for_look_left = False
_detect_played_for_look_right = False
_detect_played_for_point_left = False
_detect_played_for_point_right = False

while True:
    (
        frame_timestamp_ms,
        final_frame,
        landmark_array,
        params,
        events,
        person_bbox_coords,
        dim_mask_needed,
        confirmed_count,
        entered_count,
        last_session_all_items_confirmed,
    ) = event_detector.out_queue.get()

    manager_frame = final_frame.copy() if final_frame is not None else None

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape
        white_frame = np.full((original_h, original_w, 3), 255, dtype=np.uint8)

        # --- 【Monitor Window用】エリア定義ベースの表示切り替えロジック ---

        # 1. 判定結果表示エリア（DISPLAY_CHECK_AREA）にいる時
        if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
            # 【進入の瞬間】案内タイマーの起動と、達成状況の確定
            if not _is_notice_active and not _is_replay_active:
                _is_notice_active = True
                _display_area_notice_start_time = time.time()
                event_detector.replay_index = 0

                # 進入した時点の指差し確認の成否フラグをセッション用に保持
                _session_all_items_confirmed = (
                    all(ev["detected"] for ev in events.values()) if events else False
                )
                if not _session_all_items_confirmed and not _has_saved_exit_status:
                    event_detector.fail_count += 1
                    _has_saved_exit_status = True

            # パターンA：案内表示中の2秒間
            if _is_notice_active:
                display_frame.set_dynamic_img(
                    "Camera", white_frame, (0, 0, original_w, original_h)
                )

                if time.time() - _display_area_notice_start_time >= 2.0:
                    _is_notice_active = False
                    _is_replay_active = True

            # パターンB：案内終了後のフルスクリーンリプレイ再生
            elif _is_replay_active:
                if len(event_detector.replay_buffer) > 0:
                    idx = event_detector.replay_index % len(
                        event_detector.replay_buffer
                    )
                    replay_frame = event_detector.replay_buffer[idx].copy()
                    event_detector.replay_index += 1

                    cv.putText(
                        replay_frame,
                        "● REPLAY",
                        (20, 40),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                    display_frame.set_dynamic_img(
                        "Camera", replay_frame, (0, 0, original_w, original_h)
                    )
                else:
                    display_frame.set_dynamic_img(
                        "Camera", final_frame, (0, 0, original_w, original_h)
                    )

        # 2. 判定エリア（ROI）内で確認動作中の時
        elif event_detector.state == EventState.ENTERED_ROI:
            _is_notice_active = False
            _is_replay_active = False
            _has_saved_exit_status = False

            cv.putText(
                white_frame,
                "PLEASE PERFORM SAFE POSE CHECK (LOOK & POINT)",
                (30, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv.LINE_AA,
            )
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )

        # 3. 待機中またはそれ以外のエリア（カメラの前に誰もいない状態など）
        else:
            # ==================================================================
            # 【重要変更】エリアから退出した（NO_PERSON）瞬間、完全自動初期化
            # ==================================================================
            _is_notice_active = False
            _is_replay_active = False
            _has_saved_exit_status = False
            _ok_played_for_current_ok = False
            _buzzer_played_for_current_ng = False

            event_detector.reset()  # スコア、イベント検出フラグ、動画バッファの完全初期化
            display_frame.reset_traffic_lights()

            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )
            # ==================================================================

    else:
        display_frame.clear_dynamic_img("Camera")

    display_frame.update_display()

    # 信号機の状態更新（動作中のみ裏で動作）
    if event_detector.state == EventState.ENTERED_ROI:
        for name, event_data in events.items():
            display_frame.set_traffic_light_state(name, event_data["detected"])

    display_frame.set_person_detection_info(None, False)

    # 音声フィードバック
    if event_detector.state == EventState.ENTERED_ROI:
        for ev_name, flg_attr in [
            ("Look Left", "_detect_played_for_look_left"),
            ("Look Right", "_detect_played_for_look_right"),
            ("Point Left", "_detect_played_for_point_left"),
            ("Point Right", "_detect_played_for_point_right"),
        ]:
            if events.get(ev_name, {}).get("detected", False):
                if not locals()[flg_attr]:
                    play_detect_sound()
                    exec(f"{flg_attr} = True")
            else:
                exec(f"{flg_attr} = False")
    else:
        _detect_played_for_look_left = False
        _detect_played_for_look_right = False
        _detect_played_for_point_left = False
        _detect_played_for_point_right = False

    display_frame.set_counts(confirmed_count, event_detector.fail_count)

    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.clear_text_feedbacks()

    # --- 各画像（スタンプ・案内画像）の表示処理 ---
    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        if _is_notice_active:
            if _notice_img is not None:
                display_frame.set_result_image(
                    _notice_img,
                    (
                        (original_w - _notice_img_display_width) // 2,
                        (original_h - _notice_img_display_height) // 2,
                        _notice_img_display_width,
                        _notice_img_display_height,
                    ),
                )
        elif _is_replay_active:
            if _session_all_items_confirmed:
                display_frame.set_result_image(
                    _ok_img,
                    (
                        (original_w - _result_img_display_width) // 2,
                        (original_h - _result_img_display_height) // 2,
                        _result_img_display_width,
                        _result_img_display_height,
                    ),
                )
                if not _ok_played_for_current_ok:
                    play_ok_sound()
                    _ok_played_for_current_ok = True
                _buzzer_played_for_current_ng = False
            else:
                display_frame.set_result_image(
                    _ng_img,
                    (
                        (original_w - _result_img_display_width) // 2,
                        (original_h - _result_img_display_height) // 2,
                        _result_img_display_width,
                        _result_img_display_height,
                    ),
                )
                if not _buzzer_played_for_current_ng:
                    play_buzzer_sound()
                    _buzzer_played_for_current_ng = True
                _ok_played_for_current_ok = False

    elif event_detector.state == EventState.ENTERED_ROI:
        if _action_img is not None:
            display_frame.set_result_image(
                _action_img,
                (
                    (original_w - _action_img_display_width) // 2,
                    (original_h - _action_img_display_height) // 2,
                    _action_img_display_width,
                    _action_img_display_height,
                ),
            )

    elif event_detector.state == EventState.NO_PERSON:
        if _ankn_img is not None:
            display_frame.set_result_image(
                _ankn_img,
                (
                    (original_w - _ankn_img_display_width) // 2,
                    (original_h - _ankn_img_display_height) // 2,
                    _ankn_img_display_width,
                    _ankn_img_display_height,
                ),
            )
    else:
        display_frame.set_result_image(None)

    # 各ウィンドウへの出力
    if display_frame.display_frame is not None:
        cv.imshow("Monitor Window", display_frame.display_frame)
    if manager_frame is not None:
        cv.imshow("Manager Window", manager_frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        recognition_thread.stop()
        camera_thread.stop()
        event_detector.stop_thread()
        break
    elif key == ord("f"):
        current_prop = cv.getWindowProperty("Monitor Window", cv.WND_PROP_FULLSCREEN)
        if current_prop == cv.WINDOW_FULLSCREEN:
            cv.setWindowProperty(
                "Monitor Window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL
            )
        else:
            cv.setWindowProperty(
                "Monitor Window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
            )

cv.destroyAllWindows()
