import cv2 as cv
import mediapipe as mp
import numpy as np

from defs import project_root
from funcs.landmark_funcs import draw_landmarks_on_image, get_landmark_labels, get_pos

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

sample_image = project_root() / "sample_data" / "images" / "walking.jpg"
model_path = project_root() / "models" /  "pose_landmarker_heavy.task"
landmarks_def_path = project_root() / "models" /  "landmarks.txt"

frame = cv.imread(str(sample_image))
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

landmark_labels = get_landmark_labels(landmarks_def_path)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    num_poses=1,
    running_mode=VisionRunningMode.IMAGE,
    output_segmentation_masks=True,
)
detector = PoseLandmarker.create_from_options(options)

result = detector.detect(mp_image)
annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result.pose_landmarks)

segmentation_mask = result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255

nose = get_pos(result.pose_landmarks, landmark_labels, "nose")
print(f"Position of Nose: {nose}")

cv.imshow("annotated", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("segmentation", cv.cvtColor(visualized_mask, cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows()
