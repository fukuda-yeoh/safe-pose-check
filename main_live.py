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

# ディスプレイ解像度取得の削除
# monitor = get_monitors()[0]
# screen_width = monitor.width
# screen_height = monitor.height
# print(f"Detected screen resolution: {screen_width}x{screen_height}")

# OpenCVウィンドウをフルスクリーンに設定
cv.namedWindow("Video", cv.WINDOW_NORMAL)
cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

while True:
    (
        frame_timestamp_ms,
        final_frame,  # ここにカメラから取得した元のフレームが入ります
        landmark_array,
        params,
        events,
    ) = event_detector.out_queue.get()

    # processed_frame = None # 処理後のフレームを保持する変数は不要に

    if final_frame is not None:
        # カメラ映像の元のサイズを取得
        original_h, original_w, _ = final_frame.shape

        # ディスプレイの横幅に合わせてリサイズするロジックを削除
        # その代わりに、元のフレームサイズをそのままDisplayFrameに渡す
        display_frame.set_dynamic_img(
            "Camera", final_frame, (0, 0, original_w, original_h)
        )
    else:
        display_frame.clear_dynamic_img("Camera")  # フレームがNoneの場合はカメラ映像をクリア

    event_detected = False  # イベント発生フラグ

    # イベントクロップ画像を有効にする
    for name, event in events.items():
        if event["crop"] is not None:
            crop_h, crop_w, _ = event["crop"].shape

            # イベント画像をカメラ映像の左上と右上に表示
            # フルスクリーン表示の場合、これらの座標は画面の左上を基準にする
            # 例えば、左上のImage1は (10, 10) から (10 + crop_w, 10 + crop_h)
            # 右上のImage2は (original_w - crop_w - 10, 10) から (original_w - 10, 10 + crop_h)
            # original_w はフルスクリーン表示されたカメラ映像の幅を指す

            if name == "Look Left":  # 仮の割り当て
                display_frame.set_dynamic_img(
                    "Image1", event["crop"], (10, 10, crop_w, crop_h)
                )
            elif name == "Look Right":  # 仮の割り当て
                if final_frame is not None:  # final_frameが存在する場合にのみ計算
                    image2_x = original_w - crop_w - 10
                    display_frame.set_dynamic_img(
                        "Image2", event["crop"], (image2_x, 10, crop_w, crop_h)
                    )
                else:
                    display_frame.clear_dynamic_img("Image2")
            event_detected = True  # イベントが発生した
        else:
            # イベントが検出されなくなったら画像をクリア
            if name == "Look Left":
                display_frame.clear_dynamic_img("Image1")
            elif name == "Look Right":
                display_frame.clear_dynamic_img("Image2")

    display_frame.update_display()

    if event_detector.state == EventState.EXITED_ROI:
        event_detector.reset()
        display_frame.clear_all_dynamic_img()  # ROI外に出たら全ての動的画像をクリア

    if is_recording:
        # 録画は元のフレームサイズで行うことを想定
        recorder.record_data(
            frame_timestamp_ms, landmark_array, None, final_frame, params
        )

    # display_frame.display_frame がNoneでないことを確認
    if display_frame.display_frame is not None:
        cv.imshow(f"Video", display_frame.display_frame)
    else:
        # カメラ映像がない場合は、何もしない
        pass

    key = cv.waitKey(1)
    if key == ord("q"):
        recognition_thread.stop()
        camera_thread.stop()
        event_detector.stop_thread()
        if is_recording:
            recorder.end_record()  # 録画を終了
            is_recording = False  # 録画のフラグを終了
        # 終了時にrecognition_thread.in_queueにフレームを投入する処理は、
        # 既にスレッドが停止している可能性があるため、不要であれば削除も検討
        if final_frame is not None:
            recognition_thread.in_queue.put((frame_timestamp_ms + 1, final_frame))
        break
    elif key == ord("s"):  # 's' キーが押されたら録画を開始
        if not is_recording:
            if final_frame is not None:
                recorder.start_record(
                    f"video{record_count}", final_frame.shape
                )  # 録画を開始
                is_recording = True
                print("録画を開始しました")
            else:
                print("フレームが利用できないため録画を開始できません。")
    elif key == ord("e"):  # 'e' キーが押されたら録画を終了
        if is_recording:
            recorder.end_record()  # 録画を終了
            is_recording = False
            record_count += 1
            print("録画を終了しました")

# 終了処理
cv.destroyAllWindows()
