from queue import Queue
from threading import Thread

import mediapipe as mp

from funcs.landmark_funcs import draw_landmarks_on_image

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
            num_poses=1,  # 単一人物検出に限定
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

            # 検出結果がある場合はランドマークを描画、ない場合は元のフレームをそのまま使う
            if (
                result.pose_landmarks
            ):  # result.pose_landmarks はリストのリストだが、num_poses=1なので通常は1つの人物のみ
                # draw_landmarks_on_image が result オブジェクトも受け付けるように funcs/landmark_funcs.py を修正済み
                annotated_image = draw_landmarks_on_image(
                    mp_image.numpy_view(), result  # resultオブジェクトを直接渡す
                )
                display_frame = annotated_image
            else:
                display_frame = frame  # ランドマークが検出されない場合は元のフレーム

            # EventDetector には検出結果と、描画済みのフレームを渡す
            self.out_queue.put((result, timestamp, display_frame))

    def stop(self):
        self.stop_flag = True