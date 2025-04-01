import os

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv

from components import CameraThread, RecognitionThread, RecordWorker
from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
    landmarks_to_array,
)

camera_no = 1
model_path = project_root() / "models" / "pose_landmarker_lite.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

camera_thread = CameraThread(camera_no)
recognition_thread = RecognitionThread(model_path)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "time")
record_count = 1

camera_thread.start()
recognition_thread.start()

while True:
    timestamp, frame = camera_thread.queue.get()

    recognition_thread.in_queue.put((timestamp, frame))
    result, display_frame = recognition_thread.out_queue.get()

    if pose_landmarks_list := result.pose_landmarks:
        landmark_array = landmarks_to_array(pose_landmarks_list)
    else:
        landmark_array = None

    recorder.record_data(timestamp, landmark_array, display_frame)
    cv.imshow(f"Live Stream", display_frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        recorder.end_record()
        recognition_thread.stop()
        camera_thread.stop()
        recognition_thread.in_queue.put((timestamp + 1, frame))
        break
    elif key == ord("s"):
        recorder.start_record(f"vid{record_count}", display_frame.shape)
        record_count += 1
        print("start recording!")
    elif key == ord("e"):
        recorder.end_record()
        print("end recording!")

cv.destroyAllWindows()
