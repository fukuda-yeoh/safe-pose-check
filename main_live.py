# main_live.py

import os
import threading
import json  # jsonモジュールをインポート

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

camera_thread = CameraThread(1, camera_matrix, dist_coeffs)  # カメラ番号は適宜変更
recognition_thread = RecognitionThread(model_path, camera_thread.queue)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")
record_count = 1
is_recording = False

# --- エリア定義 ---
roi_points = []
display_check_area_points = []
clear_roi_points = []

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        loaded_config = json.load(f)
        roi_points = loaded_config.get("roi_points", [])
        display_check_area_points = loaded_config.get("display_check_area_points", [])
        clear_roi_points = loaded_config.get("clear_roi_points", [])
    print(f"ROI configuration loaded from {CONFIG_FILE}")
else:
    print("ROI configuration file not found. Using default values.")
# ------------------

event_detector = EventDetector(
    landmark_labels,
    roi_points=roi_points,
    clear_roi_points=clear_roi_points,
    display_check_area_points=display_check_area_points,
)
detector_thread = Thread(
    target=event_detector.run_in_thread, args=(recognition_thread.out_queue,)
)

display_frame = DisplayFrame()
recognition_thread.start()
detector_thread.start()
camera_thread.start()

# ウィンドウを2つ定義
cv.namedWindow("Monitor Window", cv.WINDOW_NORMAL)  # 訓練者が見る用
cv.namedWindow("Manager Window", cv.WINDOW_NORMAL)  # 管理者・リアルタイム解析確認用

# --- 各種画像を読み込む ---
ok_img_path = project_root() / "OK.png"
ng_img_path = project_root() / "NG.png"
ankn_img_path = project_root() / "Area.png"
action_img_path = project_root() / "Action.png"

_ok_img = cv.imread(str(ok_img_path), cv.IMREAD_UNCHANGED)
_ng_img = cv.imread(str(ng_img_path), cv.IMREAD_UNCHANGED)
_ankn_img = cv.imread(str(ankn_img_path), cv.IMREAD_UNCHANGED)
_action_img = cv.imread(str(action_img_path), cv.IMREAD_UNCHANGED)

if _ok_img is None:
    print(f"警告: {ok_img_path} が見つからないか読み込めません。")
if _ng_img is None:
    print(f"警告: {ng_img_path} が見つからないか読み込めません。")
if _ankn_img is None:
    print(f"警告: {ankn_img_path} が見つからないか読み込めません。")
if _action_img is None:
    print(f"警告: {action_img_path} が見つからないか読み込めません。")

_result_img_display_width = 300
_result_img_display_height = 300
_ankn_img_display_width = 400
_ankn_img_display_height = 400
_action_img_display_width = 400
_action_img_display_height = 400
# ----------------------------

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

    # Manager Window用のフレームとして、描画前のfinal_frameの状態を保持
    manager_frame = final_frame.copy() if final_frame is not None else None

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape

        # ======================================================================
        # 【新機能】Monitor Window 用の完全な白背景フレームを生成
        # ======================================================================
        white_frame = np.full((original_h, original_w, 3), 255, dtype=np.uint8)

        # --- 【Monitor Window用】フェーズ別UI切り替えロジック ---
        if event_detector.state == EventState.ENTERED_ROI:
            # 【フェーズ1: 動作中】カメラ映像ではなく白背景を使用
            # 白背景の上部に行動啓発メッセージテキストを直接描画する
            cv.putText(
                white_frame,
                "PLEASE PERFORM SAFE POSE CHECK (LOOK & POINT)",
                (30, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # 白背景で見えやすくするため黒文字に変更
                2,
                cv.LINE_AA,
            )
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )

        elif event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
            # 【フェーズ2: 結果・リプレイ確認】
            if len(event_detector.replay_buffer) > 0:
                idx = event_detector.replay_index % len(event_detector.replay_buffer)
                replay_frame = event_detector.replay_buffer[idx].copy()
                event_detector.replay_index += 1

                display_w = original_w - 280
                replay_resized = cv.resize(replay_frame, (display_w, original_h))

                cv.putText(
                    replay_resized,
                    "● REPLAY",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA,
                )

                base_frame = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                base_frame[:, :display_w] = replay_resized

                display_frame.set_dynamic_img(
                    "Camera", base_frame, (0, 0, original_w, original_h)
                )
            else:
                display_frame.set_dynamic_img(
                    "Camera", final_frame, (0, 0, original_w, original_h)
                )
        else:
            # 【フェーズ3: 待機中 / 初期状態 (NO_PERSON)】
            # 待機中もカメラ映像を表示させず、完全な白背景にする
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )
        # ----------------------------------------------------------------------
    else:
        display_frame.clear_dynamic_img("Camera")

    # 画面の基本描画（白背景の上に各種指示画像を貼り付ける処理など）を更新
    display_frame.update_display()

    # 判定表示エリアにいる場合のみ、右側にスコアボードを重ね書きする
    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        display_frame.draw_score_board(event_detector.scores)

    # 信号機の状態更新ロジック (裏処理)
    if event_detector.state != EventState.ENTERED_DISPLAY_CHECK_AREA:
        for name, event_data in events.items():
            display_frame.set_traffic_light_state(name, event_data["detected"])

    # バウンディングボックス等の初期化
    display_frame.set_person_detection_info(None, False)
    if (
        event_detector.state == EventState.ENTERED_ROI
        and event_detector._previous_state != EventState.ENTERED_ROI
    ):
        _detect_played_for_look_left = False
        _detect_played_for_look_right = False
        _detect_played_for_point_left = False
        _detect_played_for_point_right = False

    # カウント値の表示設定
    display_frame.set_counts(confirmed_count, entered_count)

    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.clear_text_feedbacks()

    # 音声フィードバックのトリガー
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

    # --- 画像（OK / NG / 待機画像 / 行動中画像）表示制御ロジック ---
    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        cam_w, cam_h = (
            (original_w, original_h) if final_frame is not None else (640, 480)
        )
        display_w = cam_w - 280
        if last_session_all_items_confirmed:
            display_frame.set_result_image(
                _ok_img,
                (
                    (display_w - _result_img_display_width) // 2,
                    (cam_h - _result_img_display_height) // 2,
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
                    (display_w - _result_img_display_width) // 2,
                    (cam_h - _result_img_display_height) // 2,
                    _result_img_display_width,
                    _result_img_display_height,
                ),
            )
            if not _buzzer_played_for_current_ng:
                play_buzzer_sound()
                _buzzer_played_for_current_ng = True
            _ok_played_for_current_ok = False

    elif event_detector.state == EventState.ENTERED_ROI:
        # ユーザーがエリア内にいる（動作中）時、白背景の中央に Action.png を表示
        if _action_img is not None and final_frame is not None:
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
        # ユーザーがいない待機状態の時、白背景の中央に 00.ankn.png を表示
        if _ankn_img is not None and final_frame is not None:
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
        _buzzer_played_for_current_ng = False
        _ok_played_for_current_ok = False

    # リセットエリア侵入時の処理
    if event_detector.state == EventState.EXITED_ROI:
        display_frame.set_person_detection_info(None, False)
        display_frame.clear_text_feedbacks()
        display_frame.set_result_image(None)
        display_frame.reset_traffic_lights()
        event_detector.reset()
        _buzzer_played_for_current_ng = False
        _ok_played_for_current_ok = False
        _detect_played_for_look_left = False
        _detect_played_for_look_right = False
        _detect_played_for_point_left = False
        _detect_played_for_point_right = False

    # Monitor Window への描画 (白背景 + 指示画像)
    if display_frame.display_frame is not None:
        cv.imshow("Monitor Window", display_frame.display_frame)

    # Manager Window への描画 (カメラ映像 + 骨格線 + リアルタイムスコア)
    if manager_frame is not None:
        cv.imshow("Manager Window", manager_frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        recognition_thread.stop()
        camera_thread.stop()
        event_detector.stop_thread()
        if is_recording:
            recorder.end_record()
            is_recording = False
        break
    elif key == ord("s"):
        if not is_recording:
            if final_frame is not None:
                recorder.start_record(f"video{record_count}", final_frame.shape)
                is_recording = True
                print("録画を開始しました")
            else:
                print("フレームが利用できないため録画を開始できません。")
    elif key == ord("e"):
        if is_recording:
            recorder.end_record()
            is_recording = False
            record_count += 1
            print("録画を終了しました")
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
