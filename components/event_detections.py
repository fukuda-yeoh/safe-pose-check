import time
from enum import Enum
from queue import Queue
from collections import deque

import cv2 as cv
import numpy as np

from funcs.calc_score import calculate_score_l, calculate_score_r
from funcs.landmark_funcs import (
    get_landmark_labels,
    get_pos_array,
    draw_arrow_on_image,
    draw_roi_on_image,
    landmarks_to_array,
)


class EventState(Enum):
    NO_PERSON = 1
    ENTERED_ROI = 2
    ENTERED_DISPLAY_CHECK_AREA = 3


class EventDetector:
    def __init__(
        self,
        landmark_labels,
        roi_points,
        clear_roi_points=None,
        display_check_area_points=None,
    ):
        self.labels = landmark_labels
        self.roi_points = np.array(roi_points, dtype=np.float32)
        self.display_check_area_points = np.array(
            display_check_area_points, dtype=np.float32
        )
        self.crop_size = (200, 200)
        self.direction_vec = np.array([1, 0, 0])

        self.virtual_points = {
            "mid shoulder": self.get_shoulder_midpoint,
            "mid eye": self.get_eye_midpoint,
            "mid mouth": self.get_mouth_midpoint,
            "mid ankle": self.get_ankle_midpoint,
        }

        self.events = self.default_event()
        self.state = EventState.NO_PERSON
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.stop_flag = False

        self.scores = {
            "LL_score": 0,
            "LR_score": 0,
            "PL_score": 0,
            "PR_score": 0,
        }

        # リプレイ動画用バッファ (最大150フレーム=約5秒分)
        self.replay_buffer = deque(maxlen=150)
        self.replay_index = 0

        self.person_bbox = None
        self.dim_mask_needed = False

        self.entered_count = 0
        self.confirmed_count = 0
        self.fail_count = 0

        self._previous_state = EventState.NO_PERSON

        self._confirmed_for_current_entry = False
        self._all_items_confirmed_in_current_session = False
        self._last_session_all_items_confirmed = False

    def reset(self):
        self.events = self.default_event()
        self.scores = {key: 0 for key in self.scores}
        self.replay_buffer.clear()
        self.replay_index = 0
        self._confirmed_for_current_entry = False
        self._all_items_confirmed_in_current_session = False

    @staticmethod
    def default_event():
        return {
            "Look Left": {
                "param": "head",
                "landmark": "nose",
                "conditional": "30 > param > 0",
                "previous": 0,
                "crop": None,
                "detected": False,
                "display_landmark": "nose",
                "display_pos": None,
            },
            "Look Right": {
                "param": "head",
                "landmark": "nose",
                "conditional": "param > 130",
                "previous": 0,
                "crop": None,
                "detected": False,
                "display_landmark": "nose",
                "display_pos": None,
            },
            "Point Left": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and 50 > params['forearm'] > 0",
                "previous": 0,
                "crop": None,
                "detected": False,
                "display_landmark": "right wrist",
                "display_pos": None,
            },
            "Point Right": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and params['forearm'] > 130",
                "previous": 0,
                "crop": None,
                "detected": False,
                "display_landmark": "right wrist",
                "display_pos": None,
            },
        }

    def update_scores(self, params):
        if not params.get("in_roi", False):
            return

        self.scores["LL_score"] = max(
            self.scores["LL_score"], calculate_score_l(params["head"], 25, 90)
        )
        self.scores["LR_score"] = max(
            self.scores["LR_score"], calculate_score_r(params["head"], 90, 155)
        )

        if params["arm_high"] > 0.6:
            self.scores["PL_score"] = max(
                self.scores["PL_score"], calculate_score_l(params["forearm"], 25, 90)
            )
            self.scores["PR_score"] = max(
                self.scores["PR_score"], calculate_score_r(params["forearm"], 90, 155)
            )

    def run_in_thread(self, in_queue):
        while not self.stop_flag:
            result, timestamp, frame = in_queue.get()
            person_bbox_coords = None
            dim_mask_needed = False

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                person_pose_landmarks = result.pose_landmarks[0]
                landmark_array = landmarks_to_array(person_pose_landmarks)

                min_x = min([lmk.x for lmk in person_pose_landmarks])
                max_x = max([lmk.x for lmk in person_pose_landmarks])
                min_y = min([lmk.y for lmk in person_pose_landmarks])
                max_y = max([lmk.y for lmk in person_pose_landmarks])

                h, w, _ = frame.shape

                margin_x = int(w * 0.05)
                margin_y = int(h * 0.05)

                x1 = int(min_x * w) - margin_x
                y1 = int(min_y * h) - margin_y
                x2 = int(max_x * w) + margin_x
                y2 = int(max_y * h) + margin_y

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                person_bbox_coords = (x1, y1, x2, y2)

                params, frame_with_roi, events = self.run(landmark_array, frame)

                self.update_scores(params)

                # 骨格線・リアルタイム点数が完全に描き込まれたフレームをバッファに記憶（判定エリアのみ）
                if self.state == EventState.ENTERED_ROI:
                    self.replay_buffer.append(frame_with_roi.copy())

                display_frame_for_output = frame_with_roi
                dim_mask_needed = True
            else:
                frame = self.draw_roi(frame)
                frame = self.draw_display_check_area(frame)

                landmark_array = None
                params = None
                events = {}
                dim_mask_needed = False

                self.state = EventState.NO_PERSON
                display_frame_for_output = frame

            self.out_queue.put(
                (
                    timestamp,
                    display_frame_for_output,
                    landmark_array,
                    params,
                    events,
                    person_bbox_coords,
                    dim_mask_needed,
                    self.confirmed_count,
                    self.fail_count,
                    self._last_session_all_items_confirmed,
                )
            )

    def run(self, landmark_array, frame=None):
        in_roi = self.feet_inside_roi(landmark_array)
        in_display_check_area = self.feet_inside_display_check_area(landmark_array)

        self._previous_state = self.state

        if in_roi:
            self.state = EventState.ENTERED_ROI
            if self._previous_state != EventState.ENTERED_ROI:
                self._confirmed_for_current_entry = False
                self._all_items_confirmed_in_current_session = False
        elif in_display_check_area:
            self.state = EventState.ENTERED_DISPLAY_CHECK_AREA
        else:
            self.state = EventState.NO_PERSON
            self._confirmed_for_current_entry = False

        eye_vec = self.get_vector(landmark_array, "left eye", "right eye")
        down_vec = self.get_vector(landmark_array, "mid eye", "mid mouth")

        head_vec = self.cross_vector(eye_vec, down_vec)
        head_vec_angle = self.get_angle_vector(head_vec)

        forearm_vec = self.get_vector(landmark_array, "right elbow", "right wrist")
        forearm_vec_angle = self.get_angle_vector(forearm_vec)

        high = self.get_high(landmark_array, "mid ankle", "nose")
        arm_high = self.get_high(landmark_array, "mid ankle", "right wrist") / (
            high if high != 0 else 1
        )

        params = {
            "in_roi": in_roi,
            "head": head_vec_angle,
            "forearm": forearm_vec_angle,
            "arm_high": arm_high,
        }

        if self.state == EventState.ENTERED_ROI:
            if not self._confirmed_for_current_entry:
                self.check_events(params, landmark_array, frame)

                all_events_detected_this_frame = all(
                    event["detected"] for event in self.events.values()
                )
                if all_events_detected_this_frame:
                    self.confirmed_count += 1
                    self._confirmed_for_current_entry = True
                    self._all_items_confirmed_in_current_session = True

        if landmark_array is not None:
            for name, event in self.events.items():
                if event["detected"] and event.get("display_landmark"):
                    display_point = self.get_point(
                        landmark_array, event["display_landmark"]
                    )
                    event["display_pos"] = (display_point[0], display_point[1])
                else:
                    event["display_pos"] = None

        if frame is not None:
            f_h, f_w, _ = frame.shape
            frame = self.draw_roi(frame)
            frame = self.draw_display_check_area(frame)

            if landmark_array is not None:
                frame = self.draw_vector(frame, landmark_array, "nose", head_vec * 0.2)
                frame = self.draw_vector(
                    frame, landmark_array, "right elbow", forearm_vec
                )

            # ==================================================================
            # 【レイアウト修正】黒背景付き・点数連動カラーの四隅スコア表示
            # ==================================================================
            score_ll = self.scores.get("LL_score", 0)
            score_lr = self.scores.get("LR_score", 0)
            score_pl = self.scores.get("PL_score", 0)
            score_pr = self.scores.get("PR_score", 0)

            # 各隅の表示内容、描画位置、および対応するスコア値の定義
            quadrant_items = [
                # 1. 左上: Look Left
                (f"Look Left: {score_ll} pts", (15, 35), score_ll, "left"),
                # 2. 右上: Look Right
                (f"Look Right: {score_lr} pts", (f_w - 215, 35), score_lr, "right"),
                # 3. 左下: Point Left
                (f"Point Left: {score_pl} pts", (15, f_h - 20), score_pl, "left"),
                # 4. 右下: Point Right
                (
                    f"Point Right: {score_pr} pts",
                    (f_w - 215, f_h - 20),
                    score_pr,
                    "right",
                ),
            ]

            for text, pos, score, align in quadrant_items:
                # 80点以上は緑文字(BGR: 0,255,0)、80点未満は赤文字(BGR: 0,0,255)
                color = (0, 255, 0) if score >= 80 else (0, 0, 255)

                # 文字列のピクセルサイズを取得して背景ボックスの大きさを自動計算
                (text_w, text_h), baseline = cv.getTextSize(
                    text, cv.FONT_HERSHEY_SIMPLEX, 0.53, 1
                )

                box_x1 = pos[0] - 8
                box_y1 = pos[1] - text_h - 8
                box_x2 = pos[0] + text_w + 8
                box_y2 = pos[1] + 6

                # テキストの背景に黒い半透明（あるいは完全な黒）のボックスを敷く
                cv.rectangle(
                    frame, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1
                )
                # ボックスに薄いグレーの枠線を引いてデザインをさらに引き締める
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 60), 1)

                # メインのカラーテキストを描画
                cv.putText(
                    frame,
                    text,
                    pos,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.53,
                    color,
                    1,
                    cv.LINE_AA,
                )
            # ==================================================================

        return params, frame, self.events

    def reset_events_detected_state(self):
        for name in self.events:
            self.events[name]["detected"] = False
            self.events[name]["previous"] = 0
            self.events[name]["crop"] = None
            self.events[name]["display_pos"] = None

    def reset_event_detected_flags_only(self):
        for name in self.events:
            self.events[name]["detected"] = False
            self.events[name]["previous"] = 0

    def check_events(self, params, landmark_array, frame=None):
        for name, event in self.events.items():
            param = params[event["param"]]
            if eval(event["conditional"]):
                event["detected"] = True
                if abs(param) > event["previous"]:
                    event["previous"] = param
                    if frame is not None:
                        height, width, _ = frame.shape
                        center = self.get_point(landmark_array, event["landmark"])
                        if center[0] <= 1.0 and center[1] <= 1.0:
                            center[0] *= width
                            center[1] *= height
                        event["crop"] = self.crop_event(frame, center)

    def crop_event(self, frame, center):
        x_start = int(center[0] - self.crop_size[0] / 2)
        x_end = int(center[0] + self.crop_size[0] / 2)
        y_start = int(center[1] - self.crop_size[1] / 2)
        y_end = int(center[1] + self.crop_size[1] / 2)
        if x_start < 0:
            x_end += abs(x_start)
            x_start = 0
        if y_start < 0:
            y_end += abs(y_start)
            y_start = 0
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        if x_end > frame_width:
            x_start -= x_end - frame_width
            x_end = frame_width
        if y_end > frame_height:
            y_start -= y_end - frame_height
            y_end = frame_height
        cropped_frame = frame[
            int(y_start) : int(y_end),
            int(x_start) : int(x_end),
            :,
        ]
        return cropped_frame

    def feet_inside_roi(self, landmark_array):
        if landmark_array is None:
            return False
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return self.point_inside_roi(left_ankle) and self.point_inside_roi(right_ankle)

    def point_inside_roi(self, point):
        return cv.pointPolygonTest(self.roi_points, (point[0], point[1]), False) >= 0

    def feet_inside_display_check_area(self, landmark_array):
        if landmark_array is None:
            return False
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return self.point_inside_display_check_area(
            left_ankle
        ) or self.point_inside_display_check_area(right_ankle)

    def point_inside_display_check_area(self, point):
        return (
            cv.pointPolygonTest(
                self.display_check_area_points, (point[0], point[1]), False
            )
            >= 0
        )

    def get_point(self, landmark_array, name):
        if landmark_array is None:
            return np.array([0, 0, 0])

        if name in self.labels:
            return get_pos_array(landmark_array, self.labels, name)[:3].copy()
        elif name in self.virtual_points:
            return self.virtual_points[name](landmark_array)
        return np.array([0, 0, 0])

    def get_shoulder_midpoint(self, landmark_array):
        left_shoulder = self.get_point(landmark_array, "left shoulder")
        right_shoulder = self.get_point(landmark_array, "right shoulder")
        return (left_shoulder + right_shoulder) / 2

    def get_eye_midpoint(self, landmark_array):
        left_eye = self.get_point(landmark_array, "left eye")
        right_eye = self.get_point(landmark_array, "right eye")
        return (left_eye + right_eye) / 2

    def get_mouth_midpoint(self, landmark_array):
        left_mouth = self.get_point(landmark_array, "mouth (left)")
        right_mouth = self.get_point(landmark_array, "mouth (right)")
        return (left_mouth + right_mouth) / 2

    def get_ankle_midpoint(self, landmark_array):
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return (left_ankle + right_ankle) / 2

    def get_high(self, landmark_array, start, end):
        start_point = self.get_point(landmark_array, start)
        end_point = self.get_point(landmark_array, end)
        return start_point[1] - end_point[1]

    def draw_roi(self, frame):
        if self.roi_points is not None and len(self.roi_points) > 0:
            frame = draw_roi_on_image(frame, self.roi_points)
        return frame

    def draw_display_check_area(self, frame):
        if (
            self.display_check_area_points is not None
            and len(self.display_check_area_points) > 0
        ):
            frame = draw_roi_on_image(
                frame, self.display_check_area_points, color=(255, 0, 255), thickness=2
            )
        return frame

    def get_vector(self, landmark_array, start, end):
        start_point = self.get_point(landmark_array, start)
        end_point = self.get_point(landmark_array, end)
        return end_point - start_point

    def cross_vector(self, start, end):
        return np.cross(start * 10, end * 10)

    @staticmethod
    def get_angle_vector(vec):
        roi_vec = np.array([0.34813878, -0.07059366, 0.1838913])
        vec_scale = np.linalg.norm(vec)
        roi_vec_scale = np.linalg.norm(roi_vec)

        if vec_scale == 0 or roi_vec_scale == 0:
            return 0

        dot_product = np.dot(vec, roi_vec)
        cos_theta = dot_product / (vec_scale * roi_vec_scale)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta) * 180 / np.pi
        return theta

    def draw_vector(self, frame, landmark_array, start, vec, *args, **kwargs):
        start_point = self.get_point(landmark_array, start)
        return draw_arrow_on_image(
            frame, start_point[:2], (start_point[:2] + vec[:2]), *args, **kwargs
        )

    def stop_thread(self):
        self.stop_flag = True
