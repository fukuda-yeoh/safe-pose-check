import cv2 as cv
import mediapipe as mp
import serial

from components import RecordWorker, EventDetector, DisplayFrame, EventState
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

# パスの設定
sample_video = project_root() / "sample_data" / "C" / "case1" / "No1.mp4"
model_path = project_root() / "models" / "pose_landmarker_heavy.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

# 動画のキャプチャ
cap = cv.VideoCapture(str(sample_video))
fps = cap.get(cv.CAP_PROP_FPS)

# ランドマークラベルの取得
landmark_labels = get_landmark_labels(landmarks_def_path)

# PoseLandmarkerのオプション設定
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    num_poses=1,
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False,
)
recognition = PoseLandmarker.create_from_options(options)

# レコーダーの初期化
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")

event_detector = EventDetector(
    landmark_labels,
    roi_points=[(0.30, 0.53), (0.30, 0.70), (0.70, 0.70), (0.70, 0.53)],
    clear_roi_points=[(0.30, 0.70), (0.30, 1), (0.70, 1), (0.70, 0.70)],
)
person_finished = False
detection_completed = False

# フレーム表示用のオブジェクト初期化
display_frame = DisplayFrame()

# 最初のフレームを読み込み、レコーダーを開始
ret, frame = cap.read()
if not ret:
    print("動画の読み込みに失敗しました。")
    cap.release()
    exit()

recorder.start_record(sample_video.stem, frame.shape)

#  COMポートを開く
print("Open Port")
# ser = serial.Serial("COM3", 9600)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # 動画フレームの処理（ポーズ検出）
    result = recognition.detect_for_video(mp_image, frame_timestamp_ms)

    if pose_landmarks_list := result.pose_landmarks:
        final_frame = mp_image.numpy_view()
        # ポーズのランドマーク情報を配列に変換
        landmark_array = landmarks_to_array(pose_landmarks_list)
        world_landmark_array = landmarks_to_array(result.pose_world_landmarks)
        # イベント検出とROI描画を含む処理
        params, annotated_image, events = event_detector.run(
            landmark_array, final_frame
        )
        event_detector.update_scores(params)  # スコアを更新
        annotated_image = draw_landmarks_on_image(annotated_image, pose_landmarks_list)
        # 多角形ROI領域を描画
        final_frame = annotated_image.copy()

    else:
        final_frame = mp_image.numpy_view()
        # ポーズが検出されなかった場合でもROIを描画
        final_frame = event_detector.draw_roi(final_frame)
        landmark_array = None
        world_landmark_array = None
        params = None
        events = {}

    # 動画フレームを表示
    display_frame.set_dynamic_img("Camera", final_frame)

    # イベントが検出された場合の処理
    for name, event in events.items():
        if event["crop"] is not None:
            display_frame.set_dynamic_img(name, event["crop"])
            event_point = 0

    # 表示の更新
    display_frame.update_display()

    # データの記録
    recorder.record_data(
        frame_timestamp_ms,
        landmark_array,
        world_landmark_array,
        final_frame,
        params,
        event_detector.scores,
    )

    # 結果の表示
    cv.imshow(f"Video", display_frame.display_frame)

    events_performed = [x["detected"] for x in event_detector.events.values()]
    if detection_completed is False and all(events_performed):
        print("OK!")
        # ser.write(b"1")
        detection_completed = True

    if event_detector.state == EventState.EXITED_ROI and detection_completed is False:
        print("NG!")
        # ser.write(b"0")
        detection_completed = True

    if event_detector.state == EventState.EXITED_ROI and detection_completed is True:
        event_detector.reset()
        display_frame.clear_all_dynamic_img()

    if cv.waitKey(1) == ord("q"):
        break

# クリーンアップ
cv.destroyAllWindows()
recorder.end_record()
cap.release()
print("Close Port")
# ser.close()
