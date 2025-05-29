import os

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv
import numpy as np
from threading import Thread
from screeninfo import get_monitors  # 再追加

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
        [359.60126359570563, 0.0, 298.67824646996394],
        [0.0, 361.0578929901831, 250.78626701590608],
        [0.0, 0.0, 1.0],
    ]
)

dist_coeffs = np.array(
    [
        -0.046472287345238956,
        0.04685134556842647,
        -0.0653804465562841,
        0.032106609308392015,
    ],
    dtype=np.float32,
)  # 例

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

# --- ディスプレイの自動検出とウィンドウ設定の変更点 ---
monitors = get_monitors()
target_monitor = None

if len(monitors) > 1:
    # 複数モニターがある場合、メインディスプレイ以外の最も広いモニターをサイネージと仮定
    secondary_monitors = [m for m in monitors if not m.is_primary]
    if secondary_monitors:
        target_monitor = max(secondary_monitors, key=lambda m: m.width * m.height)
    else:
        # メインディスプレイしか二次モニターがない場合
        target_monitor = monitors[0]
else:
    # モニターが1つしかない場合、それをターゲットとする
    target_monitor = monitors[0]

# ターゲットモニターの幅と高さを取得
display_width = target_monitor.width
display_height = target_monitor.height
display_x = target_monitor.x
display_y = target_monitor.y

# OpenCVウィンドウをターゲットモニターに移動し、フルスクリーンに設定
cv.namedWindow("Video", cv.WINDOW_NORMAL)
cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.moveWindow("Video", display_x, display_y)

print(
    f"Displaying on monitor at ({display_x}, {display_y}) with resolution {display_width}x{display_height}"
)
# --- 変更点ここまで ---


while True:
    (
        frame_timestamp_ms,
        final_frame,  # ここにカメラから取得した元のフレームが入ります
        landmark_array,
        params,
        events,
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape

        # カメラ映像をターゲットディスプレイの解像度にリサイズ
        frame_aspect_ratio = original_w / original_h
        display_aspect_ratio = display_width / display_height

        if frame_aspect_ratio > display_aspect_ratio:
            # カメラ映像が画面より横長の場合、幅に合わせて調整
            display_camera_w = display_width
            display_camera_h = int(display_width / frame_aspect_ratio)
        else:
            # カメラ映像が画面より縦長の場合、高さに合わせて調整
            display_camera_h = display_height
            display_camera_w = int(display_height * frame_aspect_ratio)

        # 画面中央に表示されるようにオフセットを計算 (Image1/Image2の配置に影響)
        offset_x = (display_width - display_camera_w) // 2
        offset_y = (display_height - display_camera_h) // 2

        display_frame.set_dynamic_img(
            "Camera",
            final_frame,
            (offset_x, offset_y, display_camera_w, display_camera_h),
        )
    else:
        display_frame.clear_dynamic_img("Camera")

    # 各イベントの検出状態を保持するフラグ
    look_left_detected = False
    point_left_detected = False
    look_right_detected = False
    point_right_detected = False

    # イベントの検出状態を更新
    for name, event in events.items():
        if event["detected"]:
            if name == "Look Left":
                look_left_detected = True
            elif name == "Point Left":
                point_left_detected = True
            elif name == "Look Right":
                look_right_detected = True
            elif name == "Point Right":
                point_right_detected = True

    # Image1の表示条件: Look Left と Point Left の両方が検出された場合
    if look_left_detected and point_left_detected:
        image1_path = project_root() / "Image" / "Leftcheck.png"
        if image1_path.exists():
            img1 = cv.imread(str(image1_path))
            if img1 is not None:
                display_width_image1 = 200  # 例: 幅 200px
                display_height_image1 = 600  # 例: 高さ 200px
                # 画面オフセットを考慮してImage1の表示位置を調整
                display_frame.set_dynamic_img(
                    "Image1",
                    img1,
                    (
                        300,
                        1200,
                        display_width_image1,
                        display_height_image1,
                    ),
                )
            else:
                print(f"Warning: Image1 file not loaded: {image1_path}")
                display_frame.clear_dynamic_img("Image1")
        else:
            print(f"Warning: Image1 file not found: {image1_path}")
            display_frame.clear_dynamic_img("Image1")
    else:
        display_frame.clear_dynamic_img("Image1")

    # Image2の表示条件: Look Right と Point Right の両方が検出された場合
    if look_right_detected and point_right_detected:
        image2_path = project_root() / "Image" / "Rightcheck.png"
        if image2_path.exists():
            img2 = cv.imread(str(image2_path))
            if img2 is not None:
                if final_frame is not None:
                    display_width_image2 = 200  # 例: 幅 200px
                    display_height_image2 = 600  # 例: 高さ 200px
                    # 画面オフセットとカメラ映像の幅を考慮してImage2の表示位置を調整
                    image2_x = offset_x + display_camera_w - display_width_image2 - 10
                    display_frame.set_dynamic_img(
                        "Image2",
                        img2,
                        (
                            1800,
                            1200,
                            display_width_image2,
                            display_height_image2,
                        ),
                    )
                else:
                    display_frame.clear_dynamic_img("Image2")
            else:
                print(f"Warning: Image2 file not loaded: {image2_path}")
                display_frame.clear_dynamic_img("Image2")
        else:
            print(f"Warning: Image2 file not found: {image2_path}")
            display_frame.clear_dynamic_img("Image2")
    else:
        display_frame.clear_dynamic_img("Image2")

    display_frame.update_display()

    if event_detector.state == EventState.EXITED_ROI:
        event_detector.reset()
        display_frame.clear_all_dynamic_img()

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

# 終了処理
cv.destroyAllWindows()
