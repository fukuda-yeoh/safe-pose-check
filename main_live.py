import os
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
)
from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
)

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
model_path = project_root() / "models" / "pose_landmarker_lite.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

camera_thread = CameraThread(1, camera_matrix, dist_coeffs)
recognition_thread = RecognitionThread(model_path, camera_thread.queue)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")
record_count = 1
is_recording = False

event_detector = EventDetector(
    landmark_labels,
    roi_points=[(0.30, 0.53), (0.30, 0.70), (0.70, 0.70), (0.70, 0.53)],
    clear_roi_points=[(0.30, 0.70), (0.30, 1), (0.70, 1), (0.70, 0.70)],
)
detector_thread = Thread(
    target=event_detector.run_in_thread, args=(recognition_thread.out_queue,)
)

display_frame = DisplayFrame()
recognition_thread.start()
detector_thread.start()
camera_thread.start()

cv.namedWindow("Video", cv.WINDOW_NORMAL)
cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# --- 画像の読み込みは不要になりました ---
# right_check_img_path = project_root() / "Rightcheck.png"
# left_check_img_path = project_root() / "Leftcheck.png"
# right_check_img = cv.imread(str(right_check_img_path), cv.IMREAD_UNCHANGED)
# left_check_img = cv.imread(str(left_check_img_path), cv.IMREAD_UNCHANGED)
# if left_check_img is None:
#     print(f"警告: {left_check_img_path} が見つからないか読み込めません。")
# if right_check_img is None:
#     print(f"警告: {right_check_img_path} が見つからないか読み込めません。")
# ------------------------------------

# --- 固定表示位置を保持する変数 (テキスト用) ---
_fixed_look_left_pos = None  # (x, y) ピクセル座標
_fixed_look_right_pos = None
_fixed_point_left_pos = None
_fixed_point_right_pos = None
# ------------------------------

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
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape
        display_frame.set_dynamic_img(
            "Camera", final_frame, (0, 0, original_w, original_h)
        )
    else:
        display_frame.clear_dynamic_img("Camera")

    # 信号機の状態更新ロジック
    for name, event_data in events.items():
        display_frame.set_traffic_light_state(name, event_data["detected"])

    # 人物の枠と薄暗い効果の設定ロジック
    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.set_person_detection_info(person_bbox_coords, dim_mask_needed)
    else:
        display_frame.set_person_detection_info(None, False)

    # カウント値の表示設定
    display_frame.set_counts(confirmed_count, entered_count)

    # --- テキストフィードバックの表示ロジック ---
    cam_w, cam_h = (original_w, original_h) if final_frame is not None else (640, 480)

    # Look Left
    if events.get("Look Left", {}).get("detected", False):
        if _fixed_look_left_pos is None:
            display_pos = events.get("Look Left", {}).get("display_pos")
            if display_pos:
                # 鼻のランドマーク（display_pos）のピクセル座標をそのまま渡す
                # display_frame側でテキストの幅と高さを考慮して位置を調整
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_look_left_pos = (text_x, text_y)

        if _fixed_look_left_pos is not None:
            display_frame.set_text_feedback("Look Left", _fixed_look_left_pos)
        else:
            display_frame.set_text_feedback("Look Left", None)
    else:
        display_frame.set_text_feedback("Look Left", None)

    # Look Right
    if events.get("Look Right", {}).get("detected", False):
        if _fixed_look_right_pos is None:
            display_pos = events.get("Look Right", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_look_right_pos = (text_x, text_y)

        if _fixed_look_right_pos is not None:
            display_frame.set_text_feedback("Look Right", _fixed_look_right_pos)
        else:
            display_frame.set_text_feedback("Look Right", None)
    else:
        display_frame.set_text_feedback("Look Right", None)

    # Point Left
    if events.get("Point Left", {}).get("detected", False):
        if _fixed_point_left_pos is None:
            display_pos = events.get("Point Left", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_point_left_pos = (text_x, text_y)

        if _fixed_point_left_pos is not None:
            display_frame.set_text_feedback("Point Left", _fixed_point_left_pos)
        else:
            display_frame.set_text_feedback("Point Left", None)
    else:
        display_frame.set_text_feedback("Point Left", None)

    # Point Right
    if events.get("Point Right", {}).get("detected", False):
        if _fixed_point_right_pos is None:
            display_pos = events.get("Point Right", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_point_right_pos = (text_x, text_y)

        if _fixed_point_right_pos is not None:
            display_frame.set_text_feedback("Point Right", _fixed_point_right_pos)
        else:
            display_frame.set_text_feedback("Point Right", None)
    else:
        display_frame.set_text_feedback("Point Right", None)
    # ------------------------------------------

    # --- リセットエリアに侵入した場合の処理 ---
    if event_detector.state == EventState.EXITED_ROI:
        event_detector.reset()
        display_frame.set_person_detection_info(None, False)
        display_frame.clear_text_feedbacks()

        # 固定位置をリセット
        _fixed_look_left_pos = None
        _fixed_look_right_pos = None
        _fixed_point_left_pos = None
        _fixed_point_right_pos = None
    # --------------------------------------------------------------------------

    display_frame.update_display()

    if is_recording:
        recorder.record_data(
            frame_timestamp_ms, landmark_array, None, final_frame, params
        )

    if display_frame.display_frame is not None:
        cv.imshow(f"Video", display_frame.display_frame)
    else:
        pass

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

cv.destroyAllWindows()
