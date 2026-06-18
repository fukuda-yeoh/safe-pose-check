# main_live.py

import os
import threading
import json  # jsonモジュールをインポート

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
    play_buzzer_sound,
    play_ok_sound,
    play_detect_sound,
)
from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
)

# 設定ファイルのパスを定義
CONFIG_DIR = project_root() / "config"
CONFIG_FILE = CONFIG_DIR / "roi_config.json"

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

camera_thread = CameraThread(1, camera_matrix, dist_coeffs)  # カメラ番号は適宜変更
recognition_thread = RecognitionThread(model_path, camera_thread.queue)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")
record_count = 1
is_recording = False

# --- エリア定義 ---
# 設定ファイルからROIポイントを読み込む
roi_points = []
display_check_area_points = []
clear_roi_points = []

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        loaded_config = json.load(f)
        roi_points = loaded_config.get("roi_points", [])
        display_check_area_points = loaded_config.get("display_check_area_points", [])
        clear_roi_points = loaded_config.get("clear_roi_points", [])
    print(f"ROI configuration loaded from {CONFIG_FILE}")
else:
    print("ROI configuration file not found. Using default values.")
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
# cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN) # ドラッグ可能にするためコメントアウト

# --- 固定表示位置を保持する変数 (テキスト用) ---
# これらの座標は、ウィンドウのサイズに合わせて調整が必要かもしれません
_fixed_look_left_pos = (375, 150)  # 左上の判定用テキスト位置
_fixed_look_right_pos = (120, 150)  # 右上の判定用テキスト位置
_fixed_point_left_pos = (375, 250)  # 左下の判定用テキスト位置
_fixed_point_right_pos = (120, 250)  # 右下の判定用テキスト位置
# ------------------------------

# --- 各種画像を読み込む ---
ok_img_path = project_root() / "OK.png"
ng_img_path = project_root() / "NG.png"
ankn_img_path = project_root() / "Area.png"
action_img_path = project_root() / "Action.png"  # 【追加】Action.pngのパス設定

_ok_img = cv.imread(str(ok_img_path), cv.IMREAD_UNCHANGED)
_ng_img = cv.imread(str(ng_img_path), cv.IMREAD_UNCHANGED)
_ankn_img = cv.imread(str(ankn_img_path), cv.IMREAD_UNCHANGED)
_action_img = cv.imread(
    str(action_img_path), cv.IMREAD_UNCHANGED
)  # 【追加】Action.pngの読み込み

if _ok_img is None:
    print(f"警告: {ok_img_path} が見つからないか読み込めません。")
if _ng_img is None:
    print(f"警告: {ng_img_path} が見つからないか読み込めません。")
if _ankn_img is None:
    print(f"警告: {ankn_img_path} が見つからないか読み込めません。")
if _action_img is None:
    print(f"警告: {action_img_path} が見つからないか読み込めません。")  # 【追加】警告文

# 判定結果画像の表示サイズ
_result_img_display_width = 300
_result_img_display_height = 300

# 待機画面画像の表示サイズ
_ankn_img_display_width = 400
_ankn_img_display_height = 400

# 行動啓発画像の表示サイズ
_action_img_display_width = 400  # 【追加】Action.pngのサイズ指定
_action_img_display_height = 400
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
        last_session_all_items_confirmed,
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape

        # ----------------------------------------------------------------------
        # 【教育システム仕様】フェーズ別UI切り替えロジック
        # ----------------------------------------------------------------------
        if event_detector.state == EventState.ENTERED_ROI:
            # 【フェーズ1: 動作中】画面を見せないために、あえて生のカメラ映像をそのまま使用
            # 画面上部に行動啓発メッセージのみを大きく表示する
            display_frame.set_dynamic_img(
                "Camera", final_frame, (0, 0, original_w, original_h)
            )

            # メッセージ描画 (案内テキスト)
            cv.rectangle(final_frame, (0, 0), (original_w, 60), (0, 0, 0), -1)
            cv.putText(
                final_frame,
                "PLEASE PERFORM SAFE POSE CHECK (LOOK & POINT)",
                (30, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )

        elif event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
            # 【フェーズ2: 結果・リプレイ確認】
            # メモリに溜まったリプレイバッファがあれば、それをパラパラマンガ風に再生
            if len(event_detector.replay_buffer) > 0:
                idx = event_detector.replay_index % len(event_detector.replay_buffer)
                replay_frame = event_detector.replay_buffer[idx].copy()
                event_detector.replay_index += 1  # 次のフレームへ

                # 画面の左側（リプレイ動画領域）にサイズを縮小してリプレイを配置
                # スコアパネル(280px)を除いた領域にフィットさせる
                display_w = original_w - 280
                replay_resized = cv.resize(replay_frame, (display_w, original_h))

                # 「REPLAY」という透かし文字を動画内に入れる
                cv.putText(
                    replay_resized,
                    "● REPLAY",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA,
                )

                # 右側のスコアボード用に、横幅を元のサイズ(original_w)に拡張した土台を作る
                base_frame = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                base_frame[:, :display_w] = replay_resized

                display_frame.set_dynamic_img(
                    "Camera", base_frame, (0, 0, original_w, original_h)
                )
            else:
                display_frame.set_dynamic_img(
                    "Camera", final_frame, (0, 0, original_w, original_h)
                )
        else:
            # 【フェーズ3: 待機中 / 初期状態】
            display_frame.set_dynamic_img(
                "Camera", final_frame, (0, 0, original_w, original_h)
            )
        # ----------------------------------------------------------------------

    else:
        display_frame.clear_dynamic_img("Camera")

    # 画面の基本描画（枠線、画像の貼り付けなど）を更新
    display_frame.update_display()

    # === 【追加】判定表示エリアにいる場合のみ、右側にスコアボードを重ね書きする ===
    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        display_frame.draw_score_board(event_detector.scores)
    # ======================================================================

    # 信号機の状態更新ロジック (動作中は画面から隠れているので裏で更新される)
    if event_detector.state != EventState.ENTERED_DISPLAY_CHECK_AREA:
        for name, event_data in events.items():
            display_frame.set_traffic_light_state(name, event_data["detected"])

    # 人物の枠と薄暗い効果の設定ロジック (指示フェーズでは無効化してすっきりさせる)
    if event_detector.state == EventState.ENTERED_ROI:
        # 動作に集中させるため、バウンディングボックスの描画指定をOFF(None)にする
        display_frame.set_person_detection_info(None, False)
        if (
            event_detector.state == EventState.ENTERED_ROI
            and event_detector._previous_state != EventState.ENTERED_ROI
        ):
            _detect_played_for_look_left = False
            _detect_played_for_look_right = False
            _detect_played_for_point_left = False
            _detect_played_for_point_right = False
    else:
        display_frame.set_person_detection_info(None, False)

    # カウント値の表示設定
    display_frame.set_counts(confirmed_count, entered_count)

    # --- テキストフィードバックの表示とdetect.mp3再生ロジック ---
    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.clear_text_feedbacks()  # リアルタイムテキストは表示しない

    # 音声のトリガーを引くための既存のイベントチェック
    for ev_name, flg_attr in [
        ("Look Left", "_detect_played_for_look_left"),
        ("Look Right", "_detect_played_for_look_right"),
        ("Point Left", "_detect_played_for_point_left"),
        ("Point Right", "_detect_played_for_point_right"),
    ]:
        if events.get(ev_name, {}).get("detected", False):
            if not locals()[flg_attr]:
                play_detect_sound()
                exec(f"{flg_attr} = True")
        else:
            exec(f"{flg_attr} = False")

    # --- 画像（OK / NG / 待機画像 / 行動中画像）表示制御ロジック ---
    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        cam_w, cam_h = (
            (original_w, original_h) if final_frame is not None else (640, 480)
        )
        display_w = cam_w - 280  # スコアボードを除いた中央にスタンプを出す
        if last_session_all_items_confirmed:
            display_frame.set_result_image(
                _ok_img,
                (
                    (display_w - _result_img_display_width) // 2,
                    (cam_h - _result_img_display_height) // 2,
                    _result_img_display_width,
                    _result_img_display_height,
                ),
            )
            if not _ok_played_for_current_ok:
                play_ok_sound()
                _ok_played_for_current_ok = True
            _buzzer_played_for_current_ng = False
        else:
            display_frame.set_result_image(
                _ng_img,
                (
                    (display_w - _result_img_display_width) // 2,
                    (cam_h - _result_img_display_height) // 2,
                    _result_img_display_width,
                    _result_img_display_height,
                ),
            )
            if not _buzzer_played_for_current_ng:
                play_buzzer_sound()
                _buzzer_played_for_current_ng = True
            _ok_played_for_current_ok = False

    elif event_detector.state == EventState.ENTERED_ROI:
        # 【追加】ユーザーがエリア内にいる（動作中）時、Action.png を中央に表示
        if _action_img is not None and final_frame is not None:
            display_frame.set_result_image(
                _action_img,
                (
                    (original_w - _action_img_display_width) // 2,  # 中央揃えX
                    (original_h - _action_img_display_height) // 2,  # 中央揃えY
                    _action_img_display_width,
                    _action_img_display_height,
                ),
            )

    elif event_detector.state == EventState.NO_PERSON:
        # ユーザーがいない待機状態の時、00.ankn.png を中央に表示
        if _ankn_img is not None and final_frame is not None:
            display_frame.set_result_image(
                _ankn_img,
                (
                    (original_w - _ankn_img_display_width) // 2,
                    (original_h - _ankn_img_display_height) // 2,
                    _ankn_img_display_width,
                    _ankn_img_display_height,
                ),
            )
    else:
        # リセットエリアなど、その他の状態では画像を非表示にする
        display_frame.set_result_image(None)
        _buzzer_played_for_current_ng = False
        _ok_played_for_current_ok = False

    # --- リセットエリアに入った場合の処理 ---
    if event_detector.state == EventState.EXITED_ROI:
        display_frame.set_person_detection_info(None, False)
        display_frame.clear_text_feedbacks()
        display_frame.set_result_image(None)
        display_frame.reset_traffic_lights()
        event_detector.reset()  # スコアと動画バッファのクリア
        _buzzer_played_for_current_ng = False
        _ok_played_for_current_ok = False
        _detect_played_for_look_left = False
        _detect_played_for_look_right = False
        _detect_played_for_point_left = False
        _detect_played_for_point_right = False

    # OpenCVのウィンドウに最終描画フレームを反映
    if display_frame.display_frame is not None:
        cv.imshow(f"Video", display_frame.display_frame)

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
    elif key == ord("f"):
        current_prop = cv.getWindowProperty("Video", cv.WND_PROP_FULLSCREEN)
        if current_prop == cv.WINDOW_FULLSCREEN:
            cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
        else:
            cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

cv.destroyAllWindows()
