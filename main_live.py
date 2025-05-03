import os

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

# シリアル通信の設定
#  COMポートを開く
print("Open Port")
# ser = serial.Serial("COM3", 9600)
# カメラ行列と歪み係数
camera_matrix = np.array(
    [
        [
            359.60126359570563,
            0.0,
            298.67824646996394
        ],
        [
            0.0,
            361.0578929901831,
            250.78626701590608
        ],
        [
            0.0,
            0.0,
            1.0
        ],
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

while True:
    (
        frame_timestamp_ms,
        final_frame,
        landmark_array,
        params,
        events,
    ) = event_detector.out_queue.get()

    display_frame.set_dynamic_img("Camera", final_frame)
    event_detected = False  # イベント発生フラグ

    for name, event in events.items():
        if event["crop"] is not None:
            display_frame.set_dynamic_img(name, event["crop"])
            event_detected = True  # イベントが発生した

    display_frame.update_display()

    if event_detector.state == EventState.EXITED_ROI :
        event_detector.reset()
        display_frame.clear_all_dynamic_img()

    if is_recording:
        recorder.record_data(
            frame_timestamp_ms, landmark_array, None, final_frame, params
        )

    # シリアル通信: イベント発生時 "1" を送信、それ以外は "0"
    # if event_detected:
    # ser.write(b"1")
    # else:
    # ser.write(b"0")

    cv.imshow(f"Video", display_frame.display_frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        recognition_thread.stop()
        camera_thread.stop()
        event_detector.stop_thread()
        if is_recording:
            recorder.end_record()  # 録画を終了
            is_recording = False  # 録画のフラグを終了
        recognition_thread.in_queue.put((frame_timestamp_ms + 1, final_frame))
        break
    elif key == ord("s"):  # 's' キーが押されたら録画を開始
        if not is_recording:
            recorder.start_record(f"video{record_count}", final_frame.shape)  # 録画を開始
            is_recording = True
            print("録画を開始しました")
    elif key == ord("e"):  # 'e' キーが押されたら録画を終了
        if is_recording:
            recorder.end_record()  # 録画を終了
            is_recording = False
            record_count += 1
            print("録画を終了しました")

# 終了処理
cv.destroyAllWindows()
# ser.close()  # シリアルポートを閉じる
print("Close Port")
