import os

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

# カメラ行列と歪み係数の定義を削除

# カメラを開く
model_path = project_root() / "models" / "pose_landmarker_lite.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)
camera_thread = CameraThread(1)
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

# OpenCVウィンドウの設定
cv.namedWindow("Video", cv.WINDOW_NORMAL)
# 初期表示はフルスクリーンにしない
# cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# カウンタ表示用の新しいウィンドウ
cv.namedWindow("Counter Dashboard", cv.WINDOW_NORMAL)
# カウンタダッシュボードの初期サイズを設定（必要に応じて調整）
cv.resizeWindow("Counter Dashboard", 400, 200)


# フルスクリーン状態を管理するフラグ
is_fullscreen = False

# 各人物の判定状態を管理するためのフラグ
current_person_confirmed_events = {
    "Look Left": False,
    "Look Right": False,
    "Point Left": False,
    "Point Right": False,
}
current_person_judged_ok = False  # 現在判定エリアにいる人物が、そのセッションでOKと判定されたかどうか

people_confirmed_count = 0  # 確認動作を検出した回数（OKになった人数）
people_entered_count = 0  # 判定エリアに侵入した人数

# ROIに人物が侵入した瞬間に一度だけ実行される処理を管理するためのフラグ
entered_roi_flag = False  # メインROIへの新規侵入を検出するためのフラグ

while True:
    (
        frame_timestamp_ms,
        final_frame,
        landmark_array,
        params,
        events_this_frame,
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape
        display_frame.set_dynamic_img(
            "Camera", final_frame, (0, 0, original_w, original_h)
        )
    else:
        display_frame.clear_dynamic_img("Camera")

    # カウンタの表示は常時有効にする
    counter_x_pos = (original_w - 300) if final_frame is not None else 340
    # ここでの display_frame.set_counter_text() は削除されているので、
    # _draw_text_with_background を直接使う。
    # この行はコメントアウトまたは削除しておくべきだったが、
    # 前回の修正で DisplayFrame.set_counter_text が削除されたエラーが出たので、
    # この部分自体が DisplayFrame._draw_text_with_background に置き換わる形になる。

    # === 判定エリア侵入者のカウントロジック (メインROI侵入時にカウント) ===
    # event_detector.state は EventDetector スレッドから更新される
    if event_detector.state == EventState.ENTERED_ROI and not entered_roi_flag:
        # 新しい人物がROIに侵入した
        people_entered_count += 1
        # この新しい人物のイベント検出状態をリセット
        for key in current_person_confirmed_events:
            current_person_confirmed_events[key] = False
        current_person_judged_ok = False  # この人物の判定はまだ完了していない
        entered_roi_flag = True  # フラグをセット (侵入者カウント済み)

    # ROIから人がいなくなった場合のフラグのリセット
    if event_detector.state == EventState.NO_PERSON:  # Exited_ROI の条件を削除
        entered_roi_flag = False  # 次のENTERED_ROIのためにフラグをリセット
        # 人物がROIにもClear ROIにもいない場合のみOK/NG表示をクリアする
        display_frame.clear_status_text()  # この行を残す

    # === OK/NG表示とイベント検出ロジック ===
    # OK/NG表示は、人物が検出されており、かつ、ROI内かClear ROI内にいる場合にのみ行う
    # それ以外の場合はクリア
    if (
        landmark_array is None
        or "nose" not in landmark_labels
        or (
            event_detector.state != EventState.ENTERED_ROI
            and event_detector.state != EventState.EXITED_ROI
        )
    ):  # 条件を変更
        display_frame.clear_status_text()
    else:  # ランドマークが検出され、ROI内かClear ROI内にいる場合
        # このフレームで検出されたイベントを、現在の人物の累積検出イベントに統合
        for name, detected_this_frame in events_this_frame.items():
            if detected_this_frame:
                current_person_confirmed_events[name] = True

        # 現在の人物が全てのイベントを完了したかチェック
        if all(current_person_confirmed_events.values()):
            if not current_person_judged_ok:  # この人物がまだOKとしてカウントされていない場合
                people_confirmed_count += 1
                current_person_judged_ok = True  # この人物はOK状態に達したことをマーク

        # 頭の上の表示ステータスを決定
        display_status_on_head = "OK" if current_person_judged_ok else "NG"

        # 頭の位置を取得してOK/NGを表示
        nose_landmark_pos = event_detector.get_point(landmark_array, "nose")
        nose_x_px = int(nose_landmark_pos[0] * original_w)
        nose_y_px = int(nose_landmark_pos[1] * original_h)
        text_y_pos = nose_y_px - 50

        display_frame.set_status_text(display_status_on_head, (nose_x_px, text_y_pos))

    display_frame.update_display()

    # === カウンタダッシュボードの表示更新 ===
    counter_frame_height = 200
    counter_frame_width = 400
    counter_dashboard_frame = np.zeros(
        (counter_frame_height, counter_frame_width, 3), dtype=np.uint8
    )  # 黒い背景

    # 1. 確認動作を検出した回数
    display_frame._draw_text_with_background(
        counter_dashboard_frame,
        f"Confirmed: {people_confirmed_count}",
        (20, 50),
        font_scale=0.8,
        thickness=2,
        color=(0, 255, 0),
        bg_color=(0, 0, 0),
        center_x_aligned=False,
    )

    # 2. 判定エリアに侵入した人数
    display_frame._draw_text_with_background(
        counter_dashboard_frame,
        f"Entered: {people_entered_count}",
        (20, 100),
        font_scale=0.8,
        thickness=2,
        color=(255, 255, 0),
        bg_color=(0, 0, 0),
        center_x_aligned=False,
    )

    # 3. 検出精度
    detection_accuracy = 0.0
    if people_entered_count > 0:
        detection_accuracy = (people_confirmed_count / people_entered_count) * 100.0

    display_frame._draw_text_with_background(
        counter_dashboard_frame,
        f"Accuracy: {detection_accuracy:.1f}%",
        (20, 150),
        font_scale=0.8,
        thickness=2,
        color=(255, 0, 255),
        bg_color=(0, 0, 0),
        center_x_aligned=False,
    )

    cv.imshow("Counter Dashboard", counter_dashboard_frame)

    # レコーディング処理
    if is_recording:
        recorder.record_data(
            frame_timestamp_ms, landmark_array, None, final_frame, params
        )

    # メイン映像の表示
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
    elif key == ord("f"):
        if is_fullscreen:
            cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
            print("フルスクリーンを解除しました")
        else:
            cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            print("フルスクリーンに設定しました")
        is_fullscreen = not is_fullscreen
    elif key == ord("r"):  # 全てのカウンタと状態をリセットするキー
        people_confirmed_count = 0
        people_entered_count = 0
        for key in current_person_confirmed_events:
            current_person_confirmed_events[key] = False
        current_person_judged_ok = False
        entered_roi_flag = False  # ROI侵入フラグもリセット
        display_frame.clear_status_text()  # リセット時にも表示をクリア
        print("全てのカウンタと状態をリセットしました")

cv.destroyAllWindows()
