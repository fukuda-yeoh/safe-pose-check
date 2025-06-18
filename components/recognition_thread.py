from queue import Queue
from threading import Thread

import mediapipe as mp

# from funcs.landmark_funcs import draw_landmarks_on_image # 削除

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class RecognitionThread(Thread):
    def __init__(self, model_path, in_queue=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_path = model_path
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_poses=1,
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self.detector = PoseLandmarker.create_from_options(options)

        if in_queue is not None:
            self.in_queue = in_queue
        else:
            self.in_queue = Queue(maxsize=30)
        self.out_queue = Queue(maxsize=30)

        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            timestamp, frame = self.in_queue.get()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            result = self.detector.detect_for_video(mp_image, timestamp)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                # person_pose_landmarks = result.pose_landmarks[0] # 不要だが残す
                # annotated_image = draw_landmarks_on_image( # 削除
                #     mp_image.numpy_view(), person_pose_landmarks # 削除
                # ) # 削除
                display_frame = mp_image.numpy_view()  # ランドマークを描画しないので、元のフレームビューをそのまま使用
            else:
                display_frame = frame
            self.out_queue.put((result, timestamp, display_frame))

    def stop(self):
        self.stop_flag = True
