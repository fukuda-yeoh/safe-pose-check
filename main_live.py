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
from components.skeleton_visualizer import SkeletonVisualizer

from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
)

CONFIG_DIR = project_root() / "config"
CONFIG_FILE = CONFIG_DIR / "roi_config.json"

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

model_path = project_root() / "models" / "pose_landmarker_full.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

camera_thread = CameraThread(1, camera_matrix, dist_coeffs)
recognition_thread = RecognitionThread(model_path, camera_thread.queue)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")

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
visualizer = SkeletonVisualizer(width=640, height=480)

has_left_high_ok = False
has_right_high_ok = False

recognition_thread.start()
detector_thread.start()
camera_thread.start()

cv.namedWindow("Monitor Window", cv.WINDOW_NORMAL)
cv.namedWindow("Manager Window", cv.WINDOW_NORMAL)

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

_display_area_notice_start_time = 0.0
_is_notice_active = False
_is_replay_active = False
_has_saved_exit_status = False
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
        world_landmarks,
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape
        white_frame = np.full((original_h, original_w, 3), 255, dtype=np.uint8)

        # ======================================================================
        # 【胸の高さ基準】左腕・右腕のそれぞれのリアルタイム物理高さを同時計算
        # ======================================================================
        is_left_arm_high = False
        is_right_arm_high = False

        if world_landmarks is not None:
            w_r_shoulder = world_landmarks[12]
            w_r_hip = world_landmarks[24]
            w_r_wrist = world_landmarks[16]

            w_l_shoulder = world_landmarks[11]
            w_l_hip = world_landmarks[23]
            w_l_wrist = world_landmarks[15]

            # 右側の「肩と腰の中点」のY座標を計算
            mid_y_r = (w_r_shoulder.y + w_r_hip.y) / 2.0
            is_right_arm_high = w_r_wrist.y < mid_y_r

            # 左側の「肩と腰の中点」のY座標を計算
            mid_y_l = (w_l_shoulder.y + w_l_hip.y) / 2.0
            is_left_arm_high = w_l_wrist.y < mid_y_l

        # エリア進入中（ROI）のみ、いずれかの手首が上がった動作を検知してホールド
        if event_detector.state == EventState.ENTERED_ROI:
            if events.get("Point Left", {}).get("detected", False):
                if is_left_arm_high or is_right_arm_high:
                    has_left_high_ok = True
            if events.get("Point Right", {}).get("detected", False):
                if is_left_arm_high or is_right_arm_high:
                    has_right_high_ok = True
        elif event_detector.state == EventState.NO_PERSON:
            has_left_high_ok = False
            has_right_high_ok = False

        # 瞬間的な腕上げ状態を両方渡すようにビジュアライザを更新
        manager_frame = visualizer.generate_frame(
            world_landmarks, final_frame, is_right_arm_high, is_left_arm_high
        )

        f_h, f_w, _ = manager_frame.shape
        one_third_h = visualizer.y_boundary_1

        # ① 上段 (TOP VIEW) の四隅に「角度スコア」を描画
        score_ll = event_detector.scores.get("LL_score", 0)
        score_lr = event_detector.scores.get("LR_score", 0)
        score_pl = event_detector.scores.get("PL_score", 0)
        score_pr = event_detector.scores.get("PR_score", 0)

        top_quadrant_items = [
            (f"Look Left: {score_ll} pts", (15, 35), score_ll >= 80),
            (f"Look Right: {score_lr} pts", (f_w - 215, 35), score_lr >= 80),
            (f"Point Left: {score_pl} pts", (15, one_third_h - 20), score_pl >= 80),
            (
                f"Point Right: {score_pr} pts",
                (f_w - 215, one_third_h - 20),
                score_pr >= 80,
            ),
        ]

        for text, pos, is_ok in top_quadrant_items:
            color = (0, 255, 0) if is_ok else (0, 0, 255)
            (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.53, 1)
            box_x1, box_y1 = pos[0] - 8, pos[1] - text_h - 8
            box_x2, box_y2 = pos[0] + text_w + 8, pos[1] + 6
            cv.rectangle(
                manager_frame, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1
            )
            cv.rectangle(
                manager_frame, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 60), 1
            )
            cv.putText(
                manager_frame,
                text,
                pos,
                cv.FONT_HERSHEY_SIMPLEX,
                0.53,
                color,
                1,
                cv.LINE_AA,
            )

        # ② 下段 (HIGH VIEW) の左右の四隅に「Left High / Right High」を描画
        txt_lh = f"Left High: {'OK' if has_left_high_ok else 'LOW'}"
        txt_rh = f"Right High: {'OK' if has_right_high_ok else 'LOW'}"

        high_view_y2 = f_h - 20
        high_quadrant_items = [
            (txt_lh, (15, high_view_y2), has_left_high_ok),
            (txt_rh, (f_w - 180, high_view_y2), has_right_high_ok),
        ]

        for text, pos, is_ok in high_quadrant_items:
            color = (0, 255, 0) if is_ok else (0, 0, 255)
            (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.53, 1)
            box_x1, box_y1 = pos[0] - 8, pos[1] - text_h - 8
            box_x2, box_y2 = pos[0] + text_w + 8, pos[1] + 6
            cv.rectangle(
                manager_frame, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1
            )
            cv.rectangle(
                manager_frame, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 60), 1
            )
            cv.putText(
                manager_frame,
                text,
                pos,
                cv.FONT_HERSHEY_SIMPLEX,
                0.53,
                color,
                1,
                cv.LINE_AA,
            )

        # ======================================================================
        # 💡【復活】すべての描画が完了した manager_frame をリプレイバッファに保存
        # ======================================================================
        if event_detector.state == EventState.ENTERED_ROI:
            event_detector.replay_buffer.append(manager_frame.copy())
        # ======================================================================

        # --- 【Monitor Window用】表示切り替えロジック ---
        if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
            if not _is_notice_active and not _is_replay_active:
                _is_notice_active = True
                _display_area_notice_start_time = time.time()
                event_detector.replay_index = 0

                _session_all_items_confirmed = (
                    all(ev["detected"] for ev in events.values()) if events else False
                )
                if not _session_all_items_confirmed and not _has_saved_exit_status:
                    event_detector.fail_count += 1
                    _has_saved_exit_status = True

            if _is_notice_active:
                display_frame.set_dynamic_img(
                    "Camera", white_frame, (0, 0, original_w, original_h)
                )
                if time.time() - _display_area_notice_start_time >= 2.0:
                    _is_notice_active = False
                    _is_replay_active = True

            elif _is_replay_active:
                if len(event_detector.replay_buffer) > 0:
                    idx = event_detector.replay_index % len(
                        event_detector.replay_buffer
                    )
                    replay_frame = event_detector.replay_buffer[idx].copy()
                    event_detector.replay_index += 1
                    display_frame.set_dynamic_img(
                        "Camera", replay_frame, (0, 0, original_w, original_h)
                    )
                else:
                    display_frame.set_dynamic_img(
                        "Camera", final_frame, (0, 0, original_w, original_h)
                    )

        elif event_detector.state == EventState.ENTERED_ROI:
            _is_notice_active = False
            _is_replay_active = False
            _has_saved_exit_status = False
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )

        else:
            _is_notice_active = False
            _is_replay_active = False
            _has_saved_exit_status = False
            _ok_played_for_current_ok = False
            _buzzer_played_for_current_ng = False
            event_detector.reset()
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )
    else:
        display_frame.clear_dynamic_img("Camera")
        raw_cam = (
            camera_thread.queue.get()[2] if not camera_thread.queue.empty() else None
        )
        # 💡 比率変更後のキーワード引数に適合させ、エラーを完全防止
        manager_frame = visualizer.generate_frame(
            world_landmarks=None,
            raw_camera_frame=raw_cam,
            is_left_height_ok=False,
            is_right_height_ok=False,
        )

    display_frame.update_display()
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

    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.clear_text_feedbacks()

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
            display_frame.set_result_image(None)
            if _session_all_items_confirmed:
                if not _ok_played_for_current_ok:
                    play_ok_sound()
                    _ok_played_for_current_ok = True
                _buzzer_played_for_current_ng = False
            else:
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
