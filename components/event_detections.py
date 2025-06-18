from enum import Enum
from queue import Queue

import cv2 as cv
import numpy as np

from funcs.calc_score import calculate_score_l, calculate_score_r
from funcs.landmark_funcs import (
    get_pos_array,
    draw_arrow_on_image,
    draw_roi_on_image,
    draw_clear_roi_on_image,
    draw_landmarks_on_image,
    landmarks_to_array,
)


class EventState(Enum):
    NO_PERSON = 1
    ENTERED_ROI = 2
    EXITED_ROI = 3


class EventDetector:
    def __init__(
        self,
        landmark_labels,
        roi_points,
        clear_roi_points,
    ):
        self.labels = landmark_labels
        self.roi_points = np.array(roi_points, dtype=np.float32)  # メインROIの頂点座標
        self.clear_roi_points = np.array(clear_roi_points, dtype=np.float32)
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

        self.person_bbox = None
        self.dim_mask_needed = False

        self.entered_count = 0
        self.confirmed_count = 0
        self._previous_state = EventState.NO_PERSON

        # Confirmedカウントを制御するためのフラグ
        self._confirmed_for_current_entry = False

    def reset(self):
        # カウントはリセットしない
        # self.entered_count = 0
        # self.confirmed_count = 0
        self.events = self.default_event()  # イベント検出状態はリセット
        self.scores = {key: 0 for key in self.scores}
        self.person_bbox = None
        self.dim_mask_needed = False
        self._previous_state = EventState.NO_PERSON
        self._confirmed_for_current_entry = False  # リセットエリアに出たらこのフラグをリセット

    @staticmethod
    def default_event():
        return {
            "Look Left": {
                "param": "head",
                "landmark": "nose",
                "conditional": "25 > param > 0",
                "previous": 0,
                "crop": None,
                "detected": False,
            },
            "Look Right": {
                "param": "head",
                "landmark": "nose",
                "conditional": "param > 155",
                "previous": 0,
                "crop": None,
                "detected": False,
            },
            "Point Left": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and 25 > params['forearm'] > 0",
                "previous": 0,
                "crop": None,
                "detected": False,
            },
            "Point Right": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and params['forearm'] > 155",
                "previous": 0,
                "crop": None,
                "detected": False,
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
                x1 = int(min_x * w)
                y1 = int(min_y * h)
                x2 = int(max_x * w)
                y2 = int(max_y * h)
                person_bbox_coords = (x1, y1, x2, y2)

                params, annotated_image, events = self.run(landmark_array, frame)

                self.update_scores(params)

                annotated_image = draw_landmarks_on_image(
                    annotated_image, person_pose_landmarks
                )
                annotated_image = self.draw_roi(
                    annotated_image,
                )
                frame = annotated_image
                events = self.events  # runメソッドで更新されたeventsを使用
                dim_mask_needed = True
            else:
                frame = self.draw_roi(
                    frame,
                )
                landmark_array = None
                params = None
                events = {}  # 人物がいない場合はイベントもなし
                dim_mask_needed = False
                self.state = EventState.NO_PERSON
                self._previous_state = EventState.NO_PERSON
                self._confirmed_for_current_entry = False  # 人物がいなくなったらConfirmedフラグもリセット

            self.out_queue.put(
                (
                    timestamp,
                    frame,
                    landmark_array,
                    params,
                    events,
                    person_bbox_coords,
                    dim_mask_needed,
                    self.confirmed_count,
                    self.entered_count,
                )
            )

    def run(self, landmark_array, frame=None):
        in_roi = self.feet_inside_roi(landmark_array)
        in_clear_roi = self.feet_inside_clear_roi(landmark_array)

        # 新しい状態を決定する前に、現在の状態を保存
        self._previous_state = self.state

        if in_roi:
            self.state = EventState.ENTERED_ROI
            # 以前ENTERED_ROI状態ではなかった場合のみカウント
            if self._previous_state != EventState.ENTERED_ROI:
                self.entered_count += 1
                self._confirmed_for_current_entry = False  # 新しい侵入でConfirmedフラグをリセット
        elif in_clear_roi:
            self.state = EventState.EXITED_ROI
            # リセットエリアに入ったら、Confirmedカウントフラグをリセットする
            self._confirmed_for_current_entry = False
            # ここでイベント検出状態もリセット（信号機を赤に戻すため）
            self.reset_events_detected_state()
        else:
            self.state = EventState.NO_PERSON
            # 人物がいなくなったらConfirmedカウントフラグをリセット
            self._confirmed_for_current_entry = False
            # 人物がいなくなったらイベント検出状態もリセット
            self.reset_events_detected_state()

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

        # Confirmedのカウントは、一度の進入につき一度のみ
        if self.state == EventState.ENTERED_ROI:
            # 既に今回の進入でConfirmedがカウントされていなければチェック
            if not self._confirmed_for_current_entry:
                self.check_events(params, landmark_array, frame)
                all_events_detected = all(
                    event["detected"] for event in self.events.values()
                )
                if all_events_detected:
                    self.confirmed_count += 1
                    self._confirmed_for_current_entry = True  # カウント済みフラグをTrueに
                    # 全て検出されたら、イベントの状態をリセット
                    # ただし、信号機はリセットエリアまで保持するため、ここのreset_events_detected_state()は不要
                    # 必要に応じて、イベントの`detected`フラグは保持しつつ、`previous`だけリセットするなどの調整が可能
                    # 今回は、検出されたら信号機は青に保持されるので、イベントの状態自体は保持
                    pass
            else:
                # 既に今回の進入でConfirmedがカウントされている場合、イベント検出は行わない
                # ただし、信号機は保持されている状態にする
                pass
        else:
            # ROI外またはリセットエリアにいる場合、イベント検出状態をリセット
            # これは信号機を赤に戻す目的
            self.reset_events_detected_state()

        if frame is not None:
            frame = self.draw_roi(frame)
            frame = self.draw_clear_roi(frame)
            if landmark_array is not None:
                frame = self.draw_vector(frame, landmark_array, "nose", head_vec)
                frame = self.draw_vector(
                    frame, landmark_array, "right elbow", forearm_vec
                )

        return params, frame, self.events

    def reset_events_detected_state(self):
        """
        全てのイベントの'detected'フラグをFalseにリセットします。
        これは主に信号機を赤に戻す目的で使用されます。
        """
        for name in self.events:
            self.events[name]["detected"] = False
            # previousもリセットしない (累積のため)
            # self.events[name]["previous"] = 0
            self.events[name]["crop"] = None

    def check_events(self, params, landmark_array, frame=None):
        for name, event in self.events.items():
            param = params[event["param"]]
            if eval(event["conditional"]):
                event["detected"] = True
                # previousは最高値を保持する目的
                if abs(param) > event["previous"]:
                    event["previous"] = param
                    # クロップ画像は今回表示しないため、この部分は不要だが残しておく
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

    def feet_inside_clear_roi(self, landmark_array):
        if landmark_array is None:
            return False
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return self.point_inside_clear_roi(left_ankle) or self.point_inside_clear_roi(
            right_ankle
        )

    def point_inside_clear_roi(self, point):
        return (
            cv.pointPolygonTest(self.clear_roi_points, (point[0], point[1]), False) >= 0
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

    def draw_clear_roi(self, frame):
        if self.clear_roi_points is not None and len(self.clear_roi_points) > 0:
            frame = draw_clear_roi_on_image(frame, self.clear_roi_points)
        return frame

    def get_vector(self, landmark_array, start, end):
        start_point = self.get_point(landmark_array, start)
        end_point = self.get_point(landmark_array, end)
        return end_point - start_point

    def cross_vector(self, start, end):
        return np.cross(start * 10, end * 10)

    @staticmethod
    def get_angle_vector(vec):
        roi_vec = np.array([0.39939338, -0.00622076, 0.02112389])
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
