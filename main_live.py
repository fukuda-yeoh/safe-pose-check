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

# --- チェック画像を読み込む ---
right_check_img_path = project_root() / "Rightcheck.png"
left_check_img_path = project_root() / "Leftcheck.png"
# error_img_path = project_root() / "error.jpg" # 削除

right_check_img = cv.imread(str(right_check_img_path), cv.IMREAD_UNCHANGED)
left_check_img = cv.imread(str(left_check_img_path), cv.IMREAD_UNCHANGED)
# error_img = cv.imread(str(error_img_path), cv.IMREAD_UNCHANGED) # 削除

# 画像の表示サイズを定義
check_img_display_width = 150
check_img_display_height = 247
# error_img_display_width = 200 # 削除
# error_img_display_height = 200 # 削除

if left_check_img is None:
    print(f"警告: {left_check_img_path} が見つからないか読み込めません。")
if right_check_img is None:
    print(f"警告: {right_check_img_path} が見つからないか読み込めません。")
# if error_img is None: # 削除
#     print(f"警告: {error_img_path} が見つからないか読み込めません。")
# ------------------------------------

while True:
    (
        frame_timestamp_ms,
        final_frame,  # recognition_thread から、ランドマークなしのフレームが来る
        landmark_array,
        params,
        events,
        person_bbox_coords,
        dim_mask_needed,
        confirmed_count,
        entered_count,
        # last_session_both_sides_confirmed, # 削除
    ) = event_detector.out_queue.get()

    # display_frame には final_frame (認識スレッドからのオリジナル画像) を設定
    # その上に event_detector が描画した ROI やベクトルを重ねる
    # これまでのロジックでは final_frame が既に recognition_thread でアノテーション済み、
    # さらに event_detector でアノテーション済みという想定だったが、
    # 骨格点を消したので、display_frame の Camera には生のカメラ映像を渡し、
    # その後で event_detector の描画結果を別途重ねる必要はない。
    # event_detector の out_queue から受け取る final_frame は、
    # 既に event_detector 内で ROI やベクトルが描画されたものとして扱う。
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

    # --- チェック画像の表示ロジック ---
    cam_w, cam_h = (original_w, original_h) if final_frame is not None else (640, 480)

    right_img_x = 10
    right_img_y = 300
    left_img_x = cam_w - check_img_display_width - 10
    left_img_y = 300

    # Look Left かつ Point Left が検出されたらLeftcheck.pngを表示
    left_check_condition = events.get("Look Left", {}).get(
        "detected", False
    ) and events.get("Point Left", {}).get("detected", False)

    if left_check_condition:
        display_frame.set_check_image(
            "Left",
            left_check_img,
            (left_img_x, left_img_y, check_img_display_width, check_img_display_height),
        )
    else:
        display_frame.set_check_image("Left", None)

    # Look Right かつ Point Right が検出されたらRightcheck.pngを表示
    right_check_condition = events.get("Look Right", {}).get(
        "detected", False
    ) and events.get("Point Right", {}).get("detected", False)

    if right_check_condition:
        display_frame.set_check_image(
            "Right",
            right_check_img,
            (
                right_img_x,
                right_img_y,
                check_img_display_width,
                check_img_display_height,
            ),
        )
    else:
        display_frame.set_check_image("Right", None)
    # ------------------------------------------

    # --- リセットエリアに侵入した場合の処理 ---
    if event_detector.state == EventState.EXITED_ROI:
        event_detector.reset()
        display_frame.set_person_detection_info(None, False)
        display_frame.set_check_image("Left", None)
        display_frame.set_check_image("Right", None)
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
        if final_frame is not None:
            recognition_thread.in_queue.put((frame_timestamp_ms + 1, final_frame))
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
