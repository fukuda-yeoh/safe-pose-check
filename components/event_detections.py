from enum import Enum
from queue import Queue

import cv2 as cv
import numpy as np

from funcs.calc_score import calculate_score_l, calculate_score_r
from funcs.landmark_funcs import (
    # get_pos_array, # この行を削除
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
        self.crop_size = (200, 200)  # この値は使われなくなりますが、残しておきます
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

        # ROIの中心X座標は単一人物認識では不要になるため削除
        # self.roi_center_x = (self.roi_points[:, 0].min() + self.roi_points[:, 0].max()) / 2.0

    @staticmethod
    def _create_event_structure():
        return {
            "Look Left": {
                "param": "head",
                "landmark": "nose",
                "conditional": "25 > param > 0",
                # "previous": 0, # previousは不要
                "detected": False,
            },
            "Look Right": {
                "param": "head",
                "landmark": "nose",
                "conditional": "param > 155",
                # "previous": 0,
                "detected": False,
            },
            "Point Left": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and 25 > params['forearm'] > 0",
                # "previous": 0,
                "detected": False,
            },
            "Point Right": {
                "param": "arm_high",
                "landmark": "right wrist",
                "conditional": "param > 0.6 and params['forearm'] > 155",
                # "previous": 0,
                "detected": False,
            },
        }

    def update_scores(self, params):
        # ROI内に存在するか
        if not params.get("in_roi", False):
            return

        # headのスコア計算
        self.scores["LL_score"] = max(
            self.scores["LL_score"], calculate_score_l(params["head"], 25, 90)
        )
        self.scores["LR_score"] = max(
            self.scores["LR_score"], calculate_score_r(params["head"], 90, 155)
        )

        # forearmのスコア計算
        if params["arm_high"] > 0.6:
            self.scores["PL_score"] = max(
                self.scores["PL_score"], calculate_score_l(params["forearm"], 25, 90)
            )
            self.scores["PR_score"] = max(
                self.scores["PR_score"], calculate_score_r(params["forearm"], 90, 155)
            )

    def run_in_thread(self, in_queue):
        while not self.stop_flag:
            result, timestamp, frame = in_queue.get()  # frameはRecognitionThreadで描画済みのもの

            # 各フレームの開始時にself.eventsのdetectedフラグをリセット
            # これにより、EventDetectorは常にそのフレームでの検出状況のみを報告する
            for name in self.events:
                self.events[name]["detected"] = False

            landmark_array = None
            params = None
            events_for_output = {name: False for name in self.events}  # 検出されなければ全てFalse

            if pose_landmarks_list := result.pose_landmarks:  # 単一人物なので [0] を使う
                # landmark_array をここで作成
                # num_poses=1なので、pose_landmarks_list は必ず1つのPoseLandmarksオブジェクトを含むリスト
                landmark_array = landmarks_to_array(pose_landmarks_list)

                # run メソッドは単一人物の処理を行う
                params, frame = self.run(landmark_array, frame)

                self.update_scores(params)  # スコアを更新

                # event_detections.pyではROIとClear ROI、ベクトルの描画のみを行う
                # 骨格の描画はrecognition_thread.pyで行われるため、ここから削除
                frame = self.draw_roi(frame)
                frame = self.draw_clear_roi(frame)

                # ベクトル描画
                head_vec = self.cross_vector(
                    self.get_vector(landmark_array, "left eye", "right eye"),
                    self.get_vector(landmark_array, "mid eye", "mid mouth"),
                )
                forearm_vec = self.get_vector(
                    landmark_array, "right elbow", "right wrist"
                )

                frame = self.draw_vector(frame, landmark_array, "nose", head_vec)
                frame = self.draw_vector(
                    frame, landmark_array, "right elbow", forearm_vec
                )

                events_for_output = {k: v["detected"] for k, v in self.events.items()}
            else:
                # ランドマークが検出されない場合、ROIとClear ROIだけ描画
                frame = self.draw_roi(frame)
                frame = self.draw_clear_roi(frame)
                self.state = EventState.NO_PERSON  # 人物がいなくなったことを通知

            self.out_queue.put(
                (timestamp, frame, landmark_array, params, events_for_output)
            )

    def run(self, landmark_array, frame=None):
        # メインROI内か判定
        in_roi = self.feet_inside_roi(landmark_array)
        in_clear_roi = self.feet_inside_clear_roi(landmark_array)

        # 状態遷移のロジック
        if in_roi:
            self.state = EventState.ENTERED_ROI
        elif (
            in_clear_roi and self.state == EventState.ENTERED_ROI
        ):  # ROIに入った後にクリアROIに入った場合のみEXITED
            self.state = EventState.EXITED_ROI
        elif not in_roi and not in_clear_roi:  # どちらのROIにもいない場合
            self.state = EventState.NO_PERSON
        # それ以外の場合は現在の状態を維持（例：NO_PERSONの状態でクリアROIにいる場合など）

        eye_vec = self.get_vector(landmark_array, "left eye", "right eye")
        down_vec = self.get_vector(landmark_array, "mid eye", "mid mouth")

        # head vector
        head_vec = self.cross_vector(eye_vec, down_vec)
        head_vec_angle = self.get_angle_vector(head_vec)

        # forearm vector
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

        if in_roi:  # ROI内にいる場合のみイベント検出
            self.check_events(params)

        # runメソッドはparamsとframeのみを返すように変更
        return params, frame

    def check_events(self, params):
        for name, event_def in self.events.items():
            param = params[event_def["param"]]
            if eval(event_def["conditional"]):
                event_def["detected"] = True

    def feet_inside_roi(self, landmark_array):
        """両足が指定領域内にあるかを判定"""
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return self.point_inside_roi(left_ankle) and self.point_inside_roi(right_ankle)

    def point_inside_roi(self, point):
        return cv.pointPolygonTest(self.roi_points, (point[0], point[1]), False) >= 0

    def feet_inside_clear_roi(self, landmark_array):
        """両足がクリア領域内にあるかを判定"""
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return self.point_inside_clear_roi(left_ankle) or self.point_inside_clear_roi(
            right_ankle
        )

    def point_inside_clear_roi(self, point):
        """`cv2.pointPolygonTest`により、pointがクリア領域内にあるかを判定"""
        return (
            cv.pointPolygonTest(self.clear_roi_points, (point[0], point[1]), False) >= 0
        )

    def get_point(self, landmark_array, name):
        """指定した特徴点の骨格情報を取得"""
        if name in self.labels:
            # landmark_array は [x, y, z, visibility, presence] の形式を想定
            label_index = list(self.labels.keys()).index(name)
            return landmark_array[label_index][:3].copy()  # x,y,z のみ返す
        elif name in self.virtual_points:
            return self.virtual_points[name](landmark_array)
        return np.array([0, 0, 0])  # ランドマークが見つからない場合のデフォルト値

    def get_shoulder_midpoint(self, landmark_array):
        """右肩と左肩の中点を計算"""
        left_shoulder = self.get_point(landmark_array, "left shoulder")
        right_shoulder = self.get_point(landmark_array, "right shoulder")
        return (left_shoulder + right_shoulder) / 2

    def get_eye_midpoint(self, landmark_array):
        """右目と左目の中点を計算"""
        left_eye = self.get_point(landmark_array, "left eye")
        right_eye = self.get_point(landmark_array, "right eye")
        return (left_eye + right_eye) / 2

    def get_mouth_midpoint(self, landmark_array):
        """右口と左口の中点を計算"""
        left_mouth = self.get_point(landmark_array, "mouth (left)")
        right_mouth = self.get_point(landmark_array, "mouth (right)")
        return (left_mouth + right_mouth) / 2

    def get_ankle_midpoint(self, landmark_array):
        """右口と左口の中点を計算"""
        left_ankle = self.get_point(landmark_array, "left ankle")
        right_ankle = self.get_point(landmark_array, "right ankle")
        return (left_ankle + right_ankle) / 2

    def get_high(self, landmark_array, start, end):
        """startからendのベクトルを計算"""
        start_y = self.get_point(landmark_array, start)[1]
        end_y = self.get_point(landmark_array, end)[1]
        return start_y - end_y

    def draw_roi(self, frame):
        """ROI領域を描画する関数"""
        if self.roi_points is not None and len(self.roi_points) > 0:
            frame = draw_roi_on_image(frame, self.roi_points)
        return frame

    def draw_clear_roi(self, frame):
        """CLEAR_ROI領域を描画する関数"""
        if self.clear_roi_points is not None and len(self.clear_roi_points) > 0:
            frame = draw_clear_roi_on_image(frame, self.clear_roi_points)
        return frame

    def get_vector(self, landmark_array, start, end):
        """startからendのベクトルを計算"""
        start_point = self.get_point(landmark_array, start)
        end_point = self.get_point(landmark_array, end)
        return end_point - start_point

    def cross_vector(self, start, end):
        """外積を計算"""
        return np.cross(start * 10, end * 10)

    @staticmethod
    def get_angle_vector(vec):
        """vecの水平成分を計算(正規化含む)"""
        roi_vec = np.array([0.39939338, -0.00622076, 0.02112389])
        vec_scale = np.linalg.norm(vec)
        roi_vec_scale = np.linalg.norm(roi_vec)

        if vec_scale == 0 or roi_vec_scale == 0:  # ゼロ除算を避ける
            return 0

        dot_product = np.dot(vec, roi_vec)
        cos_theta = dot_product / (vec_scale * roi_vec_scale)
        # acos の引数が [-1, 1] の範囲に収まるようにクリッピング
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta) * 180 / np.pi
        return theta

    def draw_vector(self, frame, landmark_array, start, vec, *args, **kwargs):
        """ベクトルの描画"""
        start_point = self.get_point(landmark_array, start)
        return draw_arrow_on_image(
            frame, start_point[:2], (start_point[:2] + vec[:2]), *args, **kwargs
        )

    def stop_thread(self):
        self.stop_flag = True
