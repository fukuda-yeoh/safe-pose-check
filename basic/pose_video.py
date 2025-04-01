import cv2 as cv
import mediapipe as mp

from components.recorder import RecordWorker
from defs import project_root
from funcs.landmark_funcs import (
    draw_landmarks_on_image,
    get_landmark_labels,
    landmarks_to_array,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

sample_video = project_root() / "sample_data" / "videos" / "No1_OK.mp4"
model_path = project_root() / "models" / "pose_landmarker_full.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

cap = cv.VideoCapture(str(sample_video))
fps = cap.get(cv.CAP_PROP_FPS)

landmark_labels = get_landmark_labels(landmarks_def_path)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    num_poses=1,
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False,
)
detector = PoseLandmarker.create_from_options(options)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "time")

ret, frame = cap.read()
recorder.start_record(sample_video.stem, frame.shape)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if pose_landmarks_list := result.pose_landmarks:
        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), pose_landmarks_list
        )
        landmark_array = landmarks_to_array(pose_landmarks_list)
        display_frame = annotated_image
    else:
        display_frame = mp_image.numpy_view()
        landmark_array = None

    recorder.record_data(frame_timestamp_ms, landmark_array, display_frame)
    cv.imshow(f"Video", display_frame)

    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
recorder.end_record()
