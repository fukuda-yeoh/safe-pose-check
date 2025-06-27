import os
import time
import threading

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
    DisplayFrame,play_buzzer_sound, play_ok_sound, play_detect_sound
)
from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
)

# カメラ行列と歪み係数
camera_matrix = np.array(
    [
        [635.8283543606941, 0.0, 253.67538054709416],
        [0.0, 636.5077974749544, 320.7767679600789],
        [0.0, 0.0, 1.0],
    ]
)

dist_coeffs = np.array(
    [
        -0.013855030664346785,
        0.036480883659782824,
        -0.004879021042728567,
        0.0019079058483680378,
        -0.21054703261551563,
    ],
    dtype=np.float32,
)

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

# --- エリア定義 ---
roi_points = [(0, 0.645), (0, 0.79), (0.713, 0.633), (0.525, 0.55)]
display_check_area_points = [
    (0.15, 0.755),  # 左上
    (0.23, 0.86),  # 左下
    (0.79, 0.72),  # 右下
    (0.65, 0.645),  # 右上
]
clear_roi_points = [
    (0.23, 0.86),  # 左上
    (0.325, 1.00),  # 左下
    (1, 1),
    (1, 0.817),  # 右下
    (0.79, 0.72),  # 右上
]
# ------------------

event_detector = EventDetector(
    landmark_labels,
    roi_points=roi_points,
    clear_roi_points=clear_roi_points,
    display_check_area_points=display_check_area_points,
)
detector_thread = Thread(
    target=event_detector.run_in_thread, args=(recognition_thread.out_queue,)
)

display_frame = DisplayFrame()
recognition_thread.start()
detector_thread.start()
camera_thread.start()

cv.namedWindow("Video", cv.WINDOW_NORMAL)
# cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# --- 固定表示位置を保持する変数 (テキスト用) ---
# これらの座標は、ウィンドウのサイズに合わせて調整が必要かもしれません
_fixed_look_left_pos = (375, 150)  # 左上の判定用テキスト位置
_fixed_look_right_pos = (120, 150)  # 右上の判定用テキスト位置
_fixed_point_left_pos = (375, 250)  # 左下の判定用テキスト位置
_fixed_point_right_pos = (120, 250)  # 右下の判定用テキスト位置
# ------------------------------

# --- 判定結果画像関連の変数 ---
_ok_img = None
_ng_img = None

# OK/NG画像を読み込む
ok_img_path = project_root() / "OK.png"
ng_img_path = project_root() / "NG.png"
_ok_img = cv.imread(str(ok_img_path), cv.IMREAD_UNCHANGED)
_ng_img = cv.imread(str(ng_img_path), cv.IMREAD_UNCHANGED)

if _ok_img is None:
    print(f"警告: {ok_img_path} が見つからないか読み込めません。")
if _ng_img is None:
    print(f"警告: {ng_img_path} が見つからないか読み込めません。")

# 判定結果画像の表示サイズ
_result_img_display_width = 300
_result_img_display_height = 300
# ----------------------------


_buzzer_played_for_current_ng = False
_ok_played_for_current_ok = False
# 各判定項目ごとの再生フラグ
_detect_played_for_look_left = False
_detect_played_for_look_right = False
_detect_played_for_point_left = False
_detect_played_for_point_right = False



# ----------------------------


while True:
    (
        frame_timestamp_ms,
        final_frame,
        landmark_array,
        params,
        events,
        person_bbox_coords,
        dim_mask_needed,
        confirmed_count,
        entered_count,
        last_session_all_items_confirmed,  # event_detections.pyから取得
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape
        display_frame.set_dynamic_img(
            "Camera", final_frame, (0, 0, original_w, original_h)
        )
    else:
        display_frame.clear_dynamic_img("Camera")

    # 信号機の状態更新ロジック
    # 判定表示エリアにいる間は信号機を更新しない
    if event_detector.state != EventState.ENTERED_DISPLAY_CHECK_AREA:
        for name, event_data in events.items():
            display_frame.set_traffic_light_state(name, event_data["detected"])

    # 人物の枠と薄暗い効果の設定ロジック
    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.set_person_detection_info(person_bbox_coords, dim_mask_needed)

        # 新規追加: ENTERED_ROIになったらdetect.mp3の再生フラグをすべてリセット
        # _previous_state は EventDetectorのインスタンス変数
        # event_detector.run() メソッドの冒頭で event_detector._previous_state が更新される
        # そのため、ここでは `event_detector.state` と `event_detector._previous_state` を比較して、
        # 状態遷移を検知するのが適切
        if (
                event_detector.state == EventState.ENTERED_ROI
                and event_detector._previous_state != EventState.ENTERED_ROI
        ):  # ROIに新しく侵入した場合
            _detect_played_for_look_left = False
            _detect_played_for_look_right = False
            _detect_played_for_point_left = False
            _detect_played_for_point_right = False

    else:
        display_frame.set_person_detection_info(None, False)

    # カウント値の表示設定
    display_frame.set_counts(confirmed_count, entered_count)

    # --- テキストフィードバックの表示とdetect.mp3再生ロジック ---
    cam_w, cam_h = (original_w, original_h) if final_frame is not None else (640, 480)

    # Look Left
    is_look_left_detected = events.get("Look Left", {}).get("detected", False)
    if is_look_left_detected:
        if _fixed_look_left_pos is None:
            display_pos = events.get("Look Left", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_look_left_pos = (text_x, text_y)

        if _fixed_look_left_pos is not None:
            display_frame.set_text_feedback("Look Left", _fixed_look_left_pos)
            if not _detect_played_for_look_left:
                play_detect_sound()
                _detect_played_for_look_left = True
        # else: fixed_posがNoneでテキスト表示できていない場合
        # do nothing with flag here, let the fixed_pos be set first
    else:  # イベントが検出されなくなった場合
        display_frame.set_text_feedback("Look Left", None)
        _detect_played_for_look_left = False

    # Look Right
    is_look_right_detected = events.get("Look Right", {}).get("detected", False)
    if is_look_right_detected:
        if _fixed_look_right_pos is None:
            display_pos = events.get("Look Right", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_look_right_pos = (text_x, text_y)

        if _fixed_look_right_pos is not None:
            display_frame.set_text_feedback("Look Right", _fixed_look_right_pos)
            if not _detect_played_for_look_right:
                play_detect_sound()
                _detect_played_for_look_right = True
        # else:
        # do nothing with flag here
    else:
        display_frame.set_text_feedback("Look Right", None)
        _detect_played_for_look_right = False

    # Point Left
    is_point_left_detected = events.get("Point Left", {}).get("detected", False)
    if is_point_left_detected:
        if _fixed_point_left_pos is None:
            display_pos = events.get("Point Left", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_point_left_pos = (text_x, text_y)

        if _fixed_point_left_pos is not None:
            display_frame.set_text_feedback("Point Left", _fixed_point_left_pos)
            if not _detect_played_for_point_left:
                play_detect_sound()
                _detect_played_for_point_left = True
        # else:
        # do nothing with flag here
    else:
        display_frame.set_text_feedback("Point Left", None)
        _detect_played_for_point_left = False

    # Point Right
    is_point_right_detected = events.get("Point Right", {}).get("detected", False)
    if is_point_right_detected:
        if _fixed_point_right_pos is None:
            display_pos = events.get("Point Right", {}).get("display_pos")
            if display_pos:
                text_x = int(display_pos[0] * cam_w)
                text_y = int(display_pos[1] * cam_h)
                _fixed_point_right_pos = (text_x, text_y)

        if _fixed_point_right_pos is not None:
            display_frame.set_text_feedback("Point Right", _fixed_point_right_pos)
            if not _detect_played_for_point_right:
                play_detect_sound()
                _detect_played_for_point_right = True
        # else:
        # do nothing with flag here
    else:
        display_frame.set_text_feedback("Point Right", None)
        _detect_played_for_point_right = False
    # ------------------------------------------

    # --- 判定結果画像表示ロジック ---
    # 判定表示エリアに侵入した場合
    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        if last_session_all_items_confirmed:  # OKの条件
            display_frame.set_result_image(
                _ok_img,
                (
                    (cam_w - _result_img_display_width) // 2,  # 中央揃えX
                    (cam_h - _result_img_display_height) // 2,  # 中央揃えY
                    _result_img_display_width,
                    _result_img_display_height,
                ),
            )
            # OK画像表示と同時にOK音を鳴らす (一度だけ)
            if not _ok_played_for_current_ok:
                play_ok_sound()
                _ok_played_for_current_ok = True
            _buzzer_played_for_current_ng = False  # NGブザーフラグをリセット
        else:  # NGの条件
            display_frame.set_result_image(
                _ng_img,
                (
                    (cam_w - _result_img_display_width) // 2,  # 中央揃えX
                    (cam_h - _result_img_display_height) // 2,  # 中央揃えY
                    _result_img_display_width,
                    _result_img_display_height,
                ),
            )
            # NG画像表示と同時にブザーを鳴らす (一度だけ)
            if not _buzzer_played_for_current_ng:
                play_buzzer_sound()
                _buzzer_played_for_current_ng = True
            _ok_played_for_current_ok = False  # OK音再生フラグをリセット
    else:
        # 判定表示エリアから出たら画像をクリア
        display_frame.set_result_image(None)
        _buzzer_played_for_current_ng = False  # 判定表示エリアを出たらブザーフラグをリセット
        _ok_played_for_current_ok = False  # 判定表示エリアを出たらOK音再生フラグをリセット

    # --------------------------------------------------------------------------

    # --- リセットエリアに侵入した場合の処理 ---
    if event_detector.state == EventState.EXITED_ROI:
        # main_live.pyで表示のリセットはここで行う
        display_frame.set_person_detection_info(None, False)  # 人物強調表示クリア
        display_frame.clear_text_feedbacks()  # テキストフィードバッククリア
        display_frame.set_result_image(None)  # 判定結果画像クリア
        display_frame.reset_traffic_lights()  # 信号機リセット

        # event_detectorの内部状態をリセット
        event_detector.reset()

        _buzzer_played_for_current_ng = False  # リセットエリアに入ったらブザーフラグをリセット
        _ok_played_for_current_ok = False  # リセットエリアに入ったらOK音再生フラグをリセット

        # 各判定項目ごとの再生フラグもリセット
        _detect_played_for_look_left = False
        _detect_played_for_look_right = False
        _detect_played_for_point_left = False
        _detect_played_for_point_right = False

    # --------------------------------------------------------------------------

    display_frame.update_display()

    if is_recording:
        recorder.record_data(frame_timestamp_ms, landmark_array, None, params)

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
    elif key == ord("f"):  # 'f' キーが押された場合の処理を追加
        # 現在のウィンドウプロパティを取得し、フルスクリーン状態を切り替える
        current_prop = cv.getWindowProperty("Video", cv.WND_PROP_FULLSCREEN)
        if current_prop == cv.WINDOW_FULLSCREEN:
            # フルスクリーンならノーマルに戻す
            cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
        else:
            # ノーマルならフルスクリーンにする
            cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

cv.destroyAllWindows()
