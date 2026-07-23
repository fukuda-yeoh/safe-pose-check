# main_live.py

import os
import threading
import json
import time

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv
import numpy as np
from threading import Thread
from scipy.signal import butter, filtfilt

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
from components.skeleton_visualizer import SkeletonVisualizer

from defs import project_root
from funcs.landmark_funcs import (
    get_landmark_labels,
)

CONFIG_DIR = project_root() / "config"
CONFIG_FILE = CONFIG_DIR / "roi_config.json"

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

model_path = project_root() / "models" / "pose_landmarker_full.task"
landmarks_def_path = project_root() / "models" / "landmarks.txt"
save_dir = project_root() / "record"
save_dir.mkdir(exist_ok=True)

camera_thread = CameraThread(1, camera_matrix, dist_coeffs)
recognition_thread = RecognitionThread(model_path, camera_thread.queue)

landmark_labels = get_landmark_labels(landmarks_def_path)
recorder = RecordWorker(save_dir, save_dir, landmark_labels, "frame")

roi_points = []
display_check_area_points = []

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        loaded_config = json.load(f)
        roi_points = loaded_config.get("roi_points", [])
        display_check_area_points = loaded_config.get("display_check_area_points", [])
    print(f"ROI configuration loaded from {CONFIG_FILE}")
else:
    print("ROI configuration file not found. Using default values.")

event_detector = EventDetector(
    landmark_labels,
    roi_points=roi_points,
    display_check_area_points=display_check_area_points,
)
detector_thread = Thread(
    target=event_detector.run_in_thread, args=(recognition_thread.out_queue,)
)

display_frame = DisplayFrame()
visualizer = SkeletonVisualizer(width=640, height=480)

has_left_high_ok = False
has_right_high_ok = False

recognition_thread.start()
detector_thread.start()
camera_thread.start()

cv.namedWindow("Monitor Window", cv.WINDOW_NORMAL)
cv.namedWindow("Manager Window", cv.WINDOW_NORMAL)

ok_img_path = project_root() / "OK.png"
ng_img_path = project_root() / "NG.png"
ankn_img_path = project_root() / "Area.png"
action_img_path = project_root() / "Action.png"
notice_img_path = project_root() / "Notice.png"

_ok_img = cv.imread(str(ok_img_path), cv.IMREAD_UNCHANGED)
_ng_img = cv.imread(str(ng_img_path), cv.IMREAD_UNCHANGED)
_ankn_img = cv.imread(str(ankn_img_path), cv.IMREAD_UNCHANGED)
_action_img = cv.imread(str(action_img_path), cv.IMREAD_UNCHANGED)
_notice_img = cv.imread(str(notice_img_path), cv.IMREAD_UNCHANGED)

_result_img_display_width = 300
_result_img_display_height = 300
_ankn_img_display_width = 400
_ankn_img_display_height = 400
_action_img_display_width = 400
_action_img_display_height = 400
_notice_img_display_width = 400
_notice_img_display_height = 400

_display_area_notice_start_time = 0.0
_is_notice_active = False
_is_replay_active = False
_has_saved_exit_status = False
_session_all_items_confirmed = False

_buzzer_played_for_current_ng = False
_ok_played_for_current_ok = False
_detect_played_for_look_left = False
_detect_played_for_look_right = False
_detect_played_for_point_left = False
_detect_played_for_point_right = False

REPLAY_STATE_FRONT = 0
REPLAY_STATE_TRANSITION = 1
REPLAY_STATE_TOP = 2
REPLAY_STATE_FINISHED = 3

replay_phase = REPLAY_STATE_FRONT
replay_pitch_deg = 0.0
replay_loop_count = 0
transition_start_time = 0.0

replay_world_landmarks_buffer = []


def smooth_landmarks_buffer(buffer, cutoff_freq=3.0, fs=30.0):
    """
    scipy.signal.butter & filtfilt を用い、録画バッファ内の3D関節座標（world_landmarks）に
    ローパスフィルタを適用して高周波の振動・手ぶれノイズを滑らかに除去する関数
    """
    n_frames = len(buffer)
    if n_frames < 10:  # フレーム数が少なすぎる場合はフィルタ処理をスキップ
        return buffer

    # 有効な world_landmarks を含むインデックスを取得
    valid_indices = [i for i, b in enumerate(buffer) if b[0] is not None]
    if len(valid_indices) < 10:
        return buffer

    sample_lmks = buffer[valid_indices[0]][0]
    n_lmks = len(sample_lmks)

    # (n_frames, n_lmks, 3) の時系列3D配列を作成
    coords = np.zeros((n_frames, n_lmks, 3), dtype=np.float32)

    # 補間用に初期値で埋める
    last_valid = None
    for i in range(n_frames):
        if buffer[i][0] is not None:
            last_valid = np.array(
                [[lm.x, lm.y, lm.z] for lm in buffer[i][0]], dtype=np.float32
            )
        if last_valid is not None:
            coords[i] = last_valid

    # バターワースローパスフィルタの設計 (2次ローパスフィルタ)
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq

    # フィルタ係数算出
    b_coeff, a_coeff = butter(N=2, Wn=normal_cutoff, btype="low", analog=False)

    # 各関節・各軸 (X, Y, Z) に対して双方向フィルタ(filtfilt)を適用
    smoothed_coords = np.zeros_like(coords)
    padlen = min(9, n_frames - 1)

    for j in range(n_lmks):
        for axis in range(3):
            try:
                smoothed_coords[:, j, axis] = filtfilt(
                    b_coeff, a_coeff, coords[:, j, axis], padlen=padlen
                )
            except Exception:
                smoothed_coords[:, j, axis] = coords[:, j, axis]

    # フィルタ適用後の平滑化座標でバッファオブジェクトを更新構築
    smoothed_buffer = []
    for i in range(n_frames):
        orig_lmks, cam_frame, r_ok, l_ok, scores = buffer[i]
        if orig_lmks is None:
            smoothed_buffer.append((None, cam_frame, r_ok, l_ok, scores))
            continue

        # 平滑化された座標で Landmark オブジェクトのディープコピーを作成
        new_lmks = []
        for j in range(n_lmks):
            lm = orig_lmks[j]

            # 新しい座標値を反映させたオブジェクトの生成
            # MediaPipeのLandmark互換オブジェクト構造を複製
            class SmoothedLandmark:
                def __init__(self, x, y, z, visibility, presence):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
                    self.presence = presence

            new_lmks.append(
                SmoothedLandmark(
                    smoothed_coords[i, j, 0],
                    smoothed_coords[i, j, 1],
                    smoothed_coords[i, j, 2],
                    getattr(lm, "visibility", 0.0),
                    getattr(lm, "presence", 0.0),
                )
            )
        smoothed_buffer.append((new_lmks, cam_frame, r_ok, l_ok, scores))

    return smoothed_buffer


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
        world_landmarks,
    ) = event_detector.out_queue.get()

    if final_frame is not None:
        original_h, original_w, _ = final_frame.shape
        white_frame = np.full((original_h, original_w, 3), 255, dtype=np.uint8)

        # 【胸の高さ基準】左腕・右腕のそれぞれのリアルタイム物理高さを同時計算
        is_left_arm_high = False
        is_right_arm_high = False

        if world_landmarks is not None:
            w_r_shoulder = world_landmarks[12]
            w_r_hip = world_landmarks[24]
            w_r_wrist = world_landmarks[16]

            w_l_shoulder = world_landmarks[11]
            w_l_hip = world_landmarks[23]
            w_l_wrist = world_landmarks[15]

            mid_y_r = (w_r_shoulder.y + w_r_hip.y) / 2.0
            is_right_arm_high = w_r_wrist.y < mid_y_r

            mid_y_l = (w_l_shoulder.y + w_l_hip.y) / 2.0
            is_left_arm_high = w_l_wrist.y < mid_y_l

        # エリア進入中（ROI）のみ、手首が上がった動作を検知してホールド
        if event_detector.state == EventState.ENTERED_ROI:
            if events.get("Point Left", {}).get("detected", False):
                if is_left_arm_high or is_right_arm_high:
                    has_left_high_ok = True
            if events.get("Point Right", {}).get("detected", False):
                if is_left_arm_high or is_right_arm_high:
                    has_right_high_ok = True
        elif event_detector.state == EventState.NO_PERSON:
            has_left_high_ok = False
            has_right_high_ok = False

        # マネージャーウィンドウ（リアルタイムレンダリング）
        manager_frame = visualizer.generate_animated_frame(
            world_landmarks,
            final_frame,
            replay_pitch_deg,
            is_right_arm_high,
            is_left_arm_high,
            event_detector.scores,
        )

        # 判定エリア(ROI)にいる間、データバッファへ蓄積
        if event_detector.state == EventState.ENTERED_ROI:
            replay_world_landmarks_buffer.append(
                (
                    world_landmarks,
                    final_frame.copy(),
                    is_right_arm_high,
                    is_left_arm_high,
                    event_detector.scores.copy(),
                )
            )

        # --- 【Monitor Window用】表示切り替えロジック ---
        if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
            if not _is_notice_active and not _is_replay_active:
                _is_notice_active = True
                _display_area_notice_start_time = time.time()

                # 💡【ローパスフィルタの一括適用】結果確認エリアに到達した瞬間、バッファを平滑化
                if len(replay_world_landmarks_buffer) > 0:
                    replay_world_landmarks_buffer = smooth_landmarks_buffer(
                        replay_world_landmarks_buffer, cutoff_freq=3.0, fs=30.0
                    )

                # リプレイ状態の初期化
                event_detector.replay_index_float = 0.0
                replay_phase = REPLAY_STATE_FRONT
                replay_pitch_deg = 0.0
                replay_loop_count = 0

                _session_all_items_confirmed = (
                    all(ev["detected"] for ev in events.values()) if events else False
                )
                if not _session_all_items_confirmed and not _has_saved_exit_status:
                    event_detector.fail_count += 1
                    _has_saved_exit_status = True

            if _is_notice_active:
                display_frame.set_dynamic_img(
                    "Camera", white_frame, (0, 0, original_w, original_h)
                )
                if time.time() - _display_area_notice_start_time >= 2.0:
                    _is_notice_active = False
                    _is_replay_active = True

            elif _is_replay_active:
                buf_len = len(replay_world_landmarks_buffer)
                if buf_len > 0:
                    SPEED_STEP = 0.75

                    if replay_phase == REPLAY_STATE_FRONT:
                        idx = int(event_detector.replay_index_float) % buf_len
                        event_detector.replay_index_float += SPEED_STEP

                        if event_detector.replay_index_float >= buf_len:
                            event_detector.replay_index_float = 0.0
                            replay_loop_count += 1
                            if replay_loop_count >= 2:
                                replay_phase = REPLAY_STATE_TRANSITION
                                transition_start_time = time.time()

                        (
                            curr_w_lmk,
                            curr_cam_frame,
                            curr_r_ok,
                            curr_l_ok,
                            curr_scores,
                        ) = replay_world_landmarks_buffer[idx]
                        replay_pitch_deg = 0.0

                    elif replay_phase == REPLAY_STATE_TRANSITION:
                        elapsed = time.time() - transition_start_time
                        progress = min(1.0, elapsed / 1.0)
                        replay_pitch_deg = progress * 90.0

                        (
                            curr_w_lmk,
                            curr_cam_frame,
                            curr_r_ok,
                            curr_l_ok,
                            curr_scores,
                        ) = replay_world_landmarks_buffer[-1]

                        if progress >= 1.0:
                            replay_phase = REPLAY_STATE_TOP
                            event_detector.replay_index_float = 0.0
                            replay_loop_count = 0

                    elif replay_phase == REPLAY_STATE_TOP:
                        idx = int(event_detector.replay_index_float) % buf_len
                        event_detector.replay_index_float += SPEED_STEP

                        if event_detector.replay_index_float >= buf_len:
                            event_detector.replay_index_float = 0.0
                            replay_loop_count += 1
                            if replay_loop_count >= 2:
                                replay_phase = REPLAY_STATE_FINISHED

                        (
                            curr_w_lmk,
                            curr_cam_frame,
                            curr_r_ok,
                            curr_l_ok,
                            curr_scores,
                        ) = replay_world_landmarks_buffer[idx]
                        replay_pitch_deg = 90.0

                    elif replay_phase == REPLAY_STATE_FINISHED:
                        (
                            curr_w_lmk,
                            curr_cam_frame,
                            curr_r_ok,
                            curr_l_ok,
                            curr_scores,
                        ) = replay_world_landmarks_buffer[-1]
                        replay_pitch_deg = 90.0

                    # フィルタ適用済みの平滑化フレームを描画
                    replay_frame = visualizer.generate_animated_frame(
                        curr_w_lmk,
                        curr_cam_frame,
                        replay_pitch_deg,
                        curr_r_ok,
                        curr_l_ok,
                        curr_scores,
                    )

                    display_frame.set_dynamic_img(
                        "Camera", replay_frame, (0, 0, original_w, original_h)
                    )
                else:
                    display_frame.set_dynamic_img(
                        "Camera", final_frame, (0, 0, original_w, original_h)
                    )

        elif event_detector.state == EventState.ENTERED_ROI:
            _is_notice_active = False
            _is_replay_active = False
            _has_saved_exit_status = False
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )

        else:
            _is_notice_active = False
            _is_replay_active = False
            _has_saved_exit_status = False
            _ok_played_for_current_ok = False
            _buzzer_played_for_current_ng = False

            replay_world_landmarks_buffer.clear()
            event_detector.reset()
            display_frame.set_dynamic_img(
                "Camera", white_frame, (0, 0, original_w, original_h)
            )
    else:
        display_frame.clear_dynamic_img("Camera")
        raw_cam = (
            camera_thread.queue.get()[2] if not camera_thread.queue.empty() else None
        )
        manager_frame = visualizer.generate_animated_frame(
            world_landmarks=None,
            raw_camera_frame=raw_cam,
            pitch_deg=0.0,
            is_left_height_ok=False,
            is_right_height_ok=False,
            scores=event_detector.scores,
        )

    display_frame.update_display()
    display_frame.set_person_detection_info(None, False)

    # 音声フィードバック
    if event_detector.state == EventState.ENTERED_ROI:
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
    else:
        _detect_played_for_look_left = False
        _detect_played_for_look_right = False
        _detect_played_for_point_left = False
        _detect_played_for_point_right = False

    if event_detector.state == EventState.ENTERED_ROI:
        display_frame.clear_text_feedbacks()

    if event_detector.state == EventState.ENTERED_DISPLAY_CHECK_AREA:
        if _is_notice_active:
            if _notice_img is not None:
                display_frame.set_result_image(
                    _notice_img,
                    (
                        (original_w - _notice_img_display_width) // 2,
                        (original_h - _notice_img_display_height) // 2,
                        _notice_img_display_width,
                        _notice_img_display_height,
                    ),
                )
        elif _is_replay_active:
            display_frame.set_result_image(None)
            if _session_all_items_confirmed:
                if not _ok_played_for_current_ok:
                    play_ok_sound()
                    _ok_played_for_current_ok = True
                _buzzer_played_for_current_ng = False
            else:
                if not _buzzer_played_for_current_ng:
                    play_buzzer_sound()
                    _buzzer_played_for_current_ng = True
                _ok_played_for_current_ok = False

    elif event_detector.state == EventState.ENTERED_ROI:
        if _action_img is not None:
            display_frame.set_result_image(
                _action_img,
                (
                    (original_w - _action_img_display_width) // 2,
                    (original_h - _action_img_display_height) // 2,
                    _action_img_display_width,
                    _action_img_display_height,
                ),
            )

    elif event_detector.state == EventState.NO_PERSON:
        if _ankn_img is not None:
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
        display_frame.set_result_image(None)

    if display_frame.display_frame is not None:
        cv.imshow("Monitor Window", display_frame.display_frame)
    if manager_frame is not None:
        cv.imshow("Manager Window", manager_frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        recognition_thread.stop()
        camera_thread.stop()
        event_detector.stop_thread()
        break
    elif key == ord("f"):
        current_prop = cv.getWindowProperty("Monitor Window", cv.WND_PROP_FULLSCREEN)
        if current_prop == cv.WINDOW_FULLSCREEN:
            cv.setWindowProperty(
                "Monitor Window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL
            )
        else:
            cv.setWindowProperty(
                "Monitor Window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
            )

cv.destroyAllWindows()
