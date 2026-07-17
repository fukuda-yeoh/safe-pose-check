# components/skeleton_visualizer.py

import cv2 as cv
import numpy as np


class SkeletonVisualizer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # 0.7 : 1.3 : 1.0 の比率
        self.y_boundary_1 = int(height * (0.7 / 3.0))
        self.y_boundary_2 = int(height * (2.0 / 3.0))

        self.h_top = self.y_boundary_1
        self.h_mid = self.y_boundary_2 - self.y_boundary_1
        self.h_bot = height - self.y_boundary_2

        # カラー定義
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_GRID = (35, 35, 35)
        self.COLOR_BONE = (255, 150, 0)
        self.COLOR_JOINT = (0, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GUIDE = (0, 80, 0)
        self.COLOR_BODY_FILL = (40, 30, 10)
        self.COLOR_ACTIVE = (0, 255, 0)
        self.COLOR_NG = (0, 0, 255)

        # フィードバック用のカラー
        self.COLOR_BLUE_OK = (255, 120, 0)  # 視認性の良いシアン・ブルー系
        self.COLOR_RED_TEXT = (50, 50, 255)  # 警告用の赤文字

    def create_base_canvas(self, raw_camera_frame=None):
        canvas = np.full((self.height, self.width, 3), self.COLOR_BG, dtype=np.uint8)

        for x in range(0, self.width, 50):
            cv.line(canvas, (x, 0), (x, self.y_boundary_1), self.COLOR_GRID, 1)
            cv.line(
                canvas, (x, self.y_boundary_2), (x, self.height), self.COLOR_GRID, 1
            )

        for y in range(0, self.y_boundary_1, 50):
            cv.line(canvas, (0, y), (self.width, y), self.COLOR_GRID, 1)
        for y in range(self.y_boundary_2, self.height, 50):
            cv.line(canvas, (0, y), (self.width, y), self.COLOR_GRID, 1)

        if raw_camera_frame is not None:
            resized_camera = cv.resize(raw_camera_frame, (self.width, self.h_mid))
            canvas[
                self.y_boundary_1 : self.y_boundary_2, 0 : self.width
            ] = resized_camera

        cv.line(
            canvas,
            (0, self.y_boundary_1),
            (self.width, self.y_boundary_1),
            (80, 80, 80),
            2,
        )
        cv.line(
            canvas,
            (0, self.y_boundary_2),
            (self.width, self.y_boundary_2),
            (80, 80, 80),
            2,
        )

        titles = [
            ("TOP VIEW (Look Angle)", 15, 25),
            ("ACTUAL CAMERA VIEW", 15, self.y_boundary_1 + 25),
            ("HIGH VIEW (Arm Height)", 15, self.y_boundary_2 + 25),
        ]
        for text, tx, ty in titles:
            cv.putText(
                canvas,
                text,
                (tx + 1, ty + 1),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                canvas,
                text,
                (tx, ty),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.COLOR_TEXT,
                1,
                cv.LINE_AA,
            )

        return canvas

    def draw_top_view(self, canvas, world_landmarks):
        """上段：TOP VIEW - 真左（0度）と真右（180度）に合格ガイドラインを水平に描画"""
        if world_landmarks is None:
            return
        try:
            scale = 130
            offset_x = self.width // 2
            offset_y = self.h_top // 2 + 15

            def to_pixel(pt):
                px = int(offset_x + pt.x * scale)
                py = int(offset_y - pt.z * scale)
                return (px, py)

            nose = to_pixel(world_landmarks[0])
            l_shoulder = to_pixel(world_landmarks[11])
            r_shoulder = to_pixel(world_landmarks[12])
            r_elbow = to_pixel(world_landmarks[14])
            r_wrist = to_pixel(world_landmarks[16])
            l_elbow_px = to_pixel(world_landmarks[13])
            l_wrist_px = to_pixel(world_landmarks[15])

            mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) // 2
            mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) // 2

            # 💡 【修正】合格ラインを真左（0度）と真右（180度）に、両肩の中点を結ぶ一本の水平なガイド線として描画
            cv.line(
                canvas,
                (mid_shoulder_x - 70, mid_shoulder_y - 10),
                (mid_shoulder_x + 70, mid_shoulder_y - 10),
                self.COLOR_GUIDE,
                1,
                cv.LINE_AA,
            )

            # 中央の基準正面線（細いグレー線）
            cv.line(
                canvas,
                (mid_shoulder_x, mid_shoulder_y),
                (mid_shoulder_x, mid_shoulder_y - 40),
                (50, 50, 50),
                1,
                cv.LINE_AA,
            )

            # 骨格リンク
            cv.line(canvas, l_shoulder, r_shoulder, self.COLOR_BONE, 7, cv.LINE_AA)
            cv.line(canvas, r_shoulder, r_elbow, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, r_elbow, r_wrist, self.COLOR_BONE, 3, cv.LINE_AA)
            cv.line(canvas, l_shoulder, l_elbow_px, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, l_elbow_px, l_wrist_px, self.COLOR_BONE, 3, cv.LINE_AA)

            # 頭部
            head_radius = 16
            cv.circle(
                canvas,
                (mid_shoulder_x, mid_shoulder_y - 10),
                head_radius,
                (60, 60, 60),
                -1,
                cv.LINE_AA,
            )
            cv.circle(
                canvas,
                (mid_shoulder_x, mid_shoulder_y - 10),
                head_radius,
                (120, 120, 120),
                1,
                cv.LINE_AA,
            )

            # 形状固定の首振り矢印
            vec_x = nose[0] - mid_shoulder_x
            vec_y = nose[1] - (mid_shoulder_y - 10)
            norm = np.sqrt(vec_x**2 + vec_y**2)

            # 💡 【修正】実際の判定用パラメータ（頭部角度）と同じロジックで不足角度を追従描画
            if norm > 0:
                vx = vec_x / norm
                vy = vec_y / norm
                arrow_tip = (
                    int(mid_shoulder_x + vx * 25),
                    int(mid_shoulder_y - 10 + vy * 25),
                )
                perp_x = -vy * 8
                perp_y = vx * 8
                p1 = (int(mid_shoulder_x + perp_x), int(mid_shoulder_y - 10 + perp_y))
                p2 = (int(mid_shoulder_x - perp_x), int(mid_shoulder_y - 10 - perp_y))

                poly_points = np.array([p1, p2, arrow_tip], dtype=np.int32)
                cv.fillPoly(canvas, [poly_points], (0, 255, 0), cv.LINE_AA)

                # 頭部の角度を計算（event_detector内部と100%同期するアルゴリズム）
                # 正面が 90度
                current_angle = np.degrees(np.arctan2(-vec_y, vec_x))

                # angleを [0, 180] にクランプ
                current_angle = np.clip(current_angle, 0.0, 180.0)

                # 左を向こうとしているか、右を向こうとしているかで分岐
                if current_angle > 90:
                    # 右向き判定（130度以上が合格。130度までの不足を計算）
                    diff_deg = int(max(0, 130 - current_angle))
                else:
                    # 左向き判定（30度以下が合格。30度までの不足を計算）
                    diff_deg = int(max(0, current_angle - 30))

                if diff_deg > 0:
                    # 合格判定の範囲（左30度、右130度）に達していない場合のみ、不足角度を表示
                    cv.putText(
                        canvas,
                        f"{diff_deg} deg",
                        (mid_shoulder_x + 20, mid_shoulder_y - 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        self.COLOR_RED_TEXT,
                        1,
                        cv.LINE_AA,
                    )

            for pt in [
                l_shoulder,
                r_shoulder,
                r_elbow,
                r_wrist,
                l_elbow_px,
                l_wrist_px,
            ]:
                cv.circle(canvas, pt, 3, self.COLOR_JOINT, -1, cv.LINE_AA)
        except Exception as e:
            pass

    def draw_front_view(
        self, canvas, world_landmarks, is_left_height_ok=False, is_right_height_ok=False
    ):
        """下段：HIGH VIEW - 胸の合格ラインの常時描画 ＆ 手首横への「OK / NG」の表示"""
        if world_landmarks is None:
            return
        try:
            scale = 130
            offset_x = self.width // 2
            offset_y = (self.y_boundary_2) + 105

            def to_pixel(pt):
                px = int(offset_x + pt.x * scale)
                py = int(offset_y + pt.y * scale)
                return (px, py)

            l_arm_color = self.COLOR_ACTIVE if is_right_height_ok else self.COLOR_NG
            r_arm_color = self.COLOR_ACTIVE if is_left_height_ok else self.COLOR_NG

            nose = to_pixel(world_landmarks[0])
            l_shoulder = to_pixel(world_landmarks[11])
            r_shoulder = to_pixel(world_landmarks[12])
            l_elbow = to_pixel(world_landmarks[13])
            l_wrist = to_pixel(world_landmarks[15])
            r_elbow = to_pixel(world_landmarks[14])
            r_wrist = to_pixel(world_landmarks[16])
            r_hip = to_pixel(world_landmarks[24])
            l_hip = to_pixel(world_landmarks[23])

            # HIGH VIEW合格ライン：胸の高さ（肩と腰の中点）
            mid_y_r_pixel = (r_shoulder[1] + r_hip[1]) // 2
            cv.line(
                canvas,
                (0, mid_y_r_pixel),
                (self.width, mid_y_r_pixel),
                self.COLOR_GUIDE,
                1,
                cv.LINE_AA,
            )

            # 胴体ポリゴン
            body_poly = np.array([l_shoulder, r_shoulder, r_hip, l_hip], dtype=np.int32)
            overlay_body = canvas.copy()
            cv.fillPoly(overlay_body, [body_poly], self.COLOR_BODY_FILL, cv.LINE_AA)
            cv.polylines(
                overlay_body, [body_poly], True, self.COLOR_BONE, 2, cv.LINE_AA
            )
            cv.addWeighted(overlay_body, 0.6, canvas, 0.4, 0, canvas)

            # 頭部
            cv.circle(canvas, (nose[0], nose[1] - 5), 14, (60, 60, 60), -1, cv.LINE_AA)
            cv.circle(
                canvas, (nose[0], nose[1] - 5), 14, (120, 120, 120), 1, cv.LINE_AA
            )

            # 体幹
            cv.line(canvas, l_shoulder, r_shoulder, self.COLOR_BONE, 6, cv.LINE_AA)
            cv.line(canvas, r_shoulder, r_hip, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, l_shoulder, l_hip, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, r_hip, l_hip, self.COLOR_BONE, 5, cv.LINE_AA)

            # 実際の腕
            cv.line(canvas, r_shoulder, r_elbow, r_arm_color, 5, cv.LINE_AA)
            cv.line(canvas, r_elbow, r_wrist, r_arm_color, 3, cv.LINE_AA)
            cv.line(canvas, l_shoulder, l_elbow, l_arm_color, 5, cv.LINE_AA)
            cv.line(canvas, l_elbow, l_wrist, l_arm_color, 3, cv.LINE_AA)

            for pt in [
                l_shoulder,
                r_shoulder,
                l_elbow,
                l_wrist,
                r_elbow,
                r_wrist,
                r_hip,
                l_hip,
            ]:
                cv.circle(canvas, pt, 3, self.COLOR_JOINT, -1, cv.LINE_AA)

            # 手首横への OK / NG 表示
            txt_r, color_r = (
                ("OK", self.COLOR_BLUE_OK)
                if is_left_height_ok
                else ("NG", self.COLOR_RED_TEXT)
            )
            txt_l, color_l = (
                ("OK", self.COLOR_BLUE_OK)
                if is_right_height_ok
                else ("NG", self.COLOR_RED_TEXT)
            )

            cv.putText(
                canvas,
                txt_r,
                (r_wrist[0] - 35, r_wrist[1] - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_r,
                1,
                cv.LINE_AA,
            )
            cv.putText(
                canvas,
                txt_l,
                (l_wrist[0] + 12, l_wrist[1] - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_l,
                1,
                cv.LINE_AA,
            )

        except Exception as e:
            pass

    def generate_frame(
        self,
        world_landmarks,
        raw_camera_frame=None,
        is_left_height_ok=False,
        is_right_height_ok=False,
    ):
        canvas = self.create_base_canvas(raw_camera_frame)
        self.draw_top_view(canvas, world_landmarks)
        self.draw_front_view(
            canvas, world_landmarks, is_left_height_ok, is_right_height_ok
        )
        return canvas
