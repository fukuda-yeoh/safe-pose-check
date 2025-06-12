# event_detections.py

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
        self.roi_points = np.array(roi_points, dtype=np.float32)
        self.clear_roi_points = np.array(clear_roi_points, dtype=np.float32)
        self.crop_size = (200, 200)
        self.direction_vec = np.array([1, 0, 0])

        self.virtual_points = {
            "mid shoulder": self.get_shoulder_midpoint,
            "mid eye": self.get_eye_midpoint,
            "mid mouth": self.get_mouth_midpoint,
            "mid ankle": self.get_ankle_midpoint,
        }

        self.events = self._create_event_structure()

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
        self.tracking_id = None
        self.current_person_landmark_array = None

        # ROIの中心点を計算
        self.roi_center_x = np.mean(self.roi_points[:, 0])
        self.roi_center_y = np.mean(self.roi_points[:, 1])

    @staticmethod
    def _create_event_structure():
        return {
            "Look Left": {
                "param": "head",
                "landmark": "nose",
                "conditional": "25 > param > 0",
                "previous": 0,
                "detected": False,
            },
            "Look Right": {
                "param": "head",
                "landmark": "nose",
                "conditional": "param > 155",
                "previous": 0,
                "detected": False,
            },
            "Point Left": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and 25 > params['forearm'] > 0",
                "previous": 0,
                "detected": False,
            },
            "Point Right": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and params['forearm'] > 155",
                "previous": 0,
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

            for name in self.events:
                self.events[name]["detected"] = False

            pose_landmarks_list = result.pose_landmarks

            if pose_landmarks_list:
                closest_person_idx = -1
                min_distance_to_center = float("inf")

                # まず、tracking_idが設定されている場合はその人物を優先的に探す
                if (
                    self.tracking_id is not None
                    and result.segmentation_masks is not None
                ):
                    for idx, mask in enumerate(result.segmentation_masks):
                        # tracking_idはMediaPipeのバージョンや設定によってsegmentation_masksに含まれない場合があります。
                        # もしtracking_idが取得できない場合、このロジックはスキップされます。
                        if (
                            hasattr(mask, "tracking_id")
                            and mask.tracking_id == self.tracking_id
                        ):
                            closest_person_idx = idx
                            # 既存のtracking_idの人物が見つかった場合、その人物を優先
                            break

                # tracking_idの人物が見つからない場合、またはtracking_idがNoneの場合、
                # ROIの中心に最も近い人物を探す
                if closest_person_idx == -1:
                    for idx, landmarks in enumerate(pose_landmarks_list):
                        landmark_array_temp = landmarks_to_array([landmarks])

                        # 両足首の中点の位置を取得
                        mid_ankle_point = self.get_ankle_midpoint(landmark_array_temp)

                        # ROIの中心と足首の中点とのユークリッド距離を計算
                        distance = np.sqrt(
                            (mid_ankle_point[0] - self.roi_center_x) ** 2
                            + (mid_ankle_point[1] - self.roi_center_y) ** 2
                        )

                        # ROI内にいる人物の中で、最も中心に近い人物を選ぶ
                        if (
                            self.feet_inside_roi(landmark_array_temp)
                            and distance < min_distance_to_center
                        ):
                            min_distance_to_center = distance
                            closest_person_idx = idx
                        # ROI内に人物がいない場合でも、まずはClear ROIにいる人物の中から選ぶ
                        elif (
                            not self.feet_inside_roi(landmark_array_temp)
                            and self.feet_inside_clear_roi(landmark_array_temp)
                            and distance < min_distance_to_center
                        ):
                            min_distance_to_center = distance
                            closest_person_idx = idx
                        # どちらにもいない場合は、距離のみで判断（ただし、この人物は判定対象にならない）
                        elif (
                            not self.feet_inside_roi(landmark_array_temp)
                            and not self.feet_inside_clear_roi(landmark_array_temp)
                            and distance < min_distance_to_center
                        ):
                            min_distance_to_center = distance
                            closest_person_idx = idx

                if closest_person_idx != -1:
                    selected_landmarks = pose_landmarks_list[closest_person_idx]
                    self.current_person_landmark_array = landmarks_to_array(
                        [selected_landmarks]
                    )

                    # tracking_idを更新（segmentation_masksから取得できる場合）
                    if result.segmentation_masks and hasattr(
                        result.segmentation_masks[closest_person_idx], "tracking_id"
                    ):
                        self.tracking_id = result.segmentation_masks[
                            closest_person_idx
                        ].tracking_id
                    else:
                        self.tracking_id = None

                    params, annotated_image = self.run(
                        self.current_person_landmark_array, frame
                    )
                    self.update_scores(params)

                    annotated_image = draw_landmarks_on_image(
                        annotated_image, pose_landmarks_list
                    )
                    annotated_image = self.draw_roi(annotated_image)
                    frame = annotated_image
                    events_for_output = {
                        k: v["detected"] for k, v in self.events.items()
                    }
                else:
                    frame = self.draw_roi(frame)
                    self.current_person_landmark_array = None
                    self.tracking_id = None
                    params = None
                    events_for_output = {name: False for name in self.events}
            else:
                frame = self.draw_roi(frame)
                self.current_person_landmark_array = None
                self.tracking_id = None
                params = None
                events_for_output = {name: False for name in self.events}

            self.out_queue.put(
                (
                    timestamp,
                    frame,
                    self.current_person_landmark_array,
                    params,
                    events_for_output,
                )
            )

    def run(self, landmark_array, frame=None):
        if landmark_array is None:
            params = {
                "in_roi": False,
                "head": 0,
                "forearm": 0,
                "arm_high": 0,
            }
            if frame is not None:
                frame = self.draw_roi(frame)
                frame = self.draw_clear_roi(frame)
            return params, frame

        in_roi = self.feet_inside_roi(landmark_array)
        in_clear_roi = self.feet_inside_clear_roi(landmark_array)

        if in_roi:
            self.state = EventState.ENTERED_ROI
        elif in_clear_roi and self.state == EventState.ENTERED_ROI:
            self.state = EventState.EXITED_ROI
        elif not in_roi and not in_clear_roi:
            self.state = EventState.NO_PERSON

        eye_vec = self.get_vector(landmark_array, "left eye", "right eye")
        down_vec = self.get_vector(landmark_array, "mid eye", "mid mouth")

        head_vec = self.cross_vector(eye_vec, down_vec)
        head_vec_angle = self.get_angle_vector(head_vec)

        forearm_vec = self.get_vector(landmark_array, "right elbow", "right wrist")
        forearm_vec_angle = self.get_angle_vector(forearm_vec)

        high = self.get_high(landmark_array, "mid ankle", "nose")
        arm_high = self.get_high(landmark_array, "mid ankle", "right wrist") / high

        params = {
            "in_roi": in_roi,
            "head": head_vec_angle,
            "forearm": forearm_vec_angle,
            "arm_high": arm_high,
        }

        if in_roi:
            self.check_events(params)

        if frame is not None:
            frame = self.draw_roi(frame)
            frame = self.draw_clear_roi(frame)
            frame = self.draw_vector(frame, landmark_array, "nose", head_vec)
            frame = self.draw_vector(frame, landmark_array, "right elbow", forearm_vec)

        return params, frame

    # ここから check_events メソッド
    def check_events(self, params):
        for name, event_def in self.events.items():
            param = params[event_def["param"]]
            if eval(event_def["conditional"]):
                event_def["detected"] = True

    # ここから feet_inside_roi メソッド
    def feet_inside_roi(self, landmark_array):
        if (
            landmark_array is None
            or "left ankle" not in self.labels
            or "right ankle" not in self.labels
        ):
            return False
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return self.point_inside_roi(left_ankle) and self.point_inside_roi(right_ankle)

    def point_inside_roi(self, point):
        return cv.pointPolygonTest(self.roi_points, (point[0], point[1]), False) >= 0

    def feet_inside_clear_roi(self, landmark_array):
        if (
            landmark_array is None
            or "left ankle" not in self.labels
            or "right ankle" not in self.labels
        ):
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
        if name in self.labels:
            return get_pos_array(landmark_array, self.labels, name)[:3].copy()
        elif name in self.virtual_points:
            return self.virtual_points[name](landmark_array)

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
        start_y = self.get_point(landmark_array, start)[1]
        end_y = self.get_point(landmark_array, end)[1]
        return start_y - end_y

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

        dot_product = np.dot(vec, roi_vec)
        cos_theta = dot_product / (vec_scale * roi_vec_scale)
        theta = np.arccos(cos_theta) * 180 / np.pi
        return theta

    def draw_vector(self, frame, landmark_array, start, vec, *args, **kwargs):
        start_point = self.get_point(landmark_array, start)
        return draw_arrow_on_image(
            frame, start_point[:2], (start_point[:2] + vec[:2]), *args, **kwargs
        )

    def stop_thread(self):
        self.stop_flag = True
