# components/skeleton_visualizer.py

import cv2 as cv
import numpy as np


class SkeletonVisualizer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # ======================================================================
        # 💡【比率変更】0.7 : 1.3 : 1.0 の比率に基づいて各エリアの境界 y 座標を厳密に計算
        # ======================================================================
        self.y_boundary_1 = int(height * (0.7 / 3.0))  # 上段と中段の境界 (112px)
        self.y_boundary_2 = int(height * (2.0 / 3.0))  # 中段と下段の境界 (320px)

        self.h_top = self.y_boundary_1  # 上段の高さ (112px)
        self.h_mid = self.y_boundary_2 - self.y_boundary_1  # 中段の高さ (208px)
        self.h_bot = height - self.y_boundary_2  # 下段の高さ (160px)

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

    def create_base_canvas(self, raw_camera_frame=None):
        canvas = np.full((self.height, self.width, 3), self.COLOR_BG, dtype=np.uint8)

        # 新しい比率のグリッド線描画
        for x in range(0, self.width, 50):
            cv.line(canvas, (x, 0), (x, self.y_boundary_1), self.COLOR_GRID, 1)
            cv.line(
                canvas, (x, self.y_boundary_2), (x, self.height), self.COLOR_GRID, 1
            )

        for y in range(0, self.y_boundary_1, 50):
            cv.line(canvas, (0, y), (self.width, y), self.COLOR_GRID, 1)
        for y in range(self.y_boundary_2, self.height, 50):
            cv.line(canvas, (0, y), (self.width, y), self.COLOR_GRID, 1)

        # 中段：拡大されたサイズ（高さ208px）に合わせてリサイズして埋め込み
        if raw_camera_frame is not None:
            resized_camera = cv.resize(raw_camera_frame, (self.width, self.h_mid))
            canvas[
                self.y_boundary_1 : self.y_boundary_2, 0 : self.width
            ] = resized_camera

        # 境界セパレータ線
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
        """上段：TOP VIEW - 座標を「少し上に配置」調整"""
        if world_landmarks is None:
            return
        try:
            scale = 130
            offset_x = self.width // 2

            # 💡 エリアの高さ縮小に伴い、offset_y を少し上（中央よりさらに浅い位置）へシフト
            offset_y = self.h_top // 2 + 5

            def to_pixel(pt):
                px = int(offset_x + pt.x * scale)
                py = int(offset_y - pt.z * scale)
                return (px, py)

            nose = to_pixel(world_landmarks[0])
            l_shoulder = to_pixel(world_landmarks[11])
            r_shoulder = to_pixel(world_landmarks[12])
            r_elbow = to_pixel(world_landmarks[14])
            r_wrist = to_pixel(world_landmarks[16])
            l_elbow = world_landmarks[13]
            l_wrist = world_landmarks[15]

            l_elbow_px = to_pixel(l_elbow)
            l_wrist_px = to_pixel(l_wrist)

            mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) // 2
            mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) // 2
            cv.line(
                canvas,
                (mid_shoulder_x, mid_shoulder_y),
                (mid_shoulder_x, mid_shoulder_y - 40),
                self.COLOR_GUIDE,
                1,
                cv.LINE_AA,
            )

            cv.line(canvas, l_shoulder, r_shoulder, self.COLOR_BONE, 7, cv.LINE_AA)
            cv.line(canvas, r_shoulder, r_elbow, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, r_elbow, r_wrist, self.COLOR_BONE, 3, cv.LINE_AA)
            cv.line(canvas, l_shoulder, l_elbow_px, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, l_elbow_px, l_wrist_px, self.COLOR_BONE, 3, cv.LINE_AA)

            # 形状固定型の視線三角形
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

            vec_x = nose[0] - mid_shoulder_x
            vec_y = nose[1] - (mid_shoulder_y - 10)
            norm = np.sqrt(vec_x**2 + vec_y**2)

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
        """下段：HIGH VIEW - 座標を「少し下に配置」調整"""
        if world_landmarks is None:
            return
        try:
            scale = 130
            offset_x = self.width // 2

            # 💡 基準点をさらに下げて、下段枠の下寄りに人型をどっしり配置 (+95 -> +105)
            offset_y = self.y_boundary_2 + 105

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

            cv.line(
                canvas,
                (0, r_shoulder[1]),
                (self.width, r_shoulder[1]),
                self.COLOR_GUIDE,
                1,
                cv.LINE_4,
            )

            body_poly = np.array([l_shoulder, r_shoulder, r_hip, l_hip], dtype=np.int32)
            overlay = canvas.copy()
            cv.fillPoly(overlay, [body_poly], self.COLOR_BODY_FILL, cv.LINE_AA)
            cv.polylines(overlay, [body_poly], True, self.COLOR_BONE, 2, cv.LINE_AA)
            cv.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

            cv.circle(canvas, (nose[0], nose[1] - 5), 14, (60, 60, 60), -1, cv.LINE_AA)
            cv.circle(
                canvas, (nose[0], nose[1] - 5), 14, (120, 120, 120), 1, cv.LINE_AA
            )

            cv.line(canvas, l_shoulder, r_shoulder, self.COLOR_BONE, 6, cv.LINE_AA)
            cv.line(canvas, r_shoulder, r_hip, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, l_shoulder, l_hip, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, r_hip, l_hip, self.COLOR_BONE, 5, cv.LINE_AA)

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
