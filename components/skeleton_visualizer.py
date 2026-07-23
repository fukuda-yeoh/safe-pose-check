# components/skeleton_visualizer.py

import cv2 as cv
import numpy as np


class SkeletonVisualizer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # 上下2分割（各240px）
        self.split_y = height // 2

        # カラー定義
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_GRID = (35, 35, 35)
        self.COLOR_BONE = (255, 150, 0)  # 通常の骨格色（青系）
        self.COLOR_JOINT = (0, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GUIDE = (0, 80, 0)
        self.COLOR_BODY_FILL = (40, 30, 10)
        self.COLOR_ACTIVE = (0, 255, 0)  # OK時の腕の黄緑色

        # フィードバックカラー
        self.COLOR_LIME_OK = (0, 255, 120)  # OK表示の文字色（黄緑）
        self.COLOR_RED_TEXT = (50, 50, 255)  # NG表示の文字色（赤）

    def project_3d(self, x, y, z, pitch_deg, offset_x, offset_y, scale=120):
        rad = np.radians(pitch_deg)
        cos_p = np.cos(rad)
        sin_p = np.sin(rad)
        y_rot = y * cos_p - z * sin_p

        px = int(offset_x + x * scale)
        py = int(offset_y + y_rot * scale)
        return (px, py)

    def draw_3d_skeleton(
        self,
        canvas,
        world_landmarks,
        pitch_deg=0.0,
        is_height_ok=False,
        scores=None,
        show_pop=False,  # 正面ポップアップフラグ
        pop_wrist_idx=16,  # ポップアップを表示する手首のインデックス
        show_left_score_pop=False,  # 💡 左側スコアポップアップ（Look Left, Point Left）
        show_right_score_pop=False,  # 💡 右側スコアポップアップ（Look Right, Point Right）
    ):
        if world_landmarks is None:
            return

        try:
            scale = 120
            offset_x = self.width // 2
            offset_y = int(160 * (1.0 - pitch_deg / 90.0) + 120 * (pitch_deg / 90.0))

            def get_px(idx):
                lm = world_landmarks[idx]
                return self.project_3d(
                    lm.x, lm.y, lm.z, pitch_deg, offset_x, offset_y, scale
                )

            nose = get_px(0)
            l_shoulder = get_px(11)
            r_shoulder = get_px(12)
            l_elbow = get_px(13)
            r_elbow = get_px(14)
            l_wrist = get_px(15)
            r_wrist = get_px(16)
            l_hip = get_px(23)
            r_hip = get_px(24)

            mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) // 2
            mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) // 2

            # ガイドライン描画
            if pitch_deg < 45.0:
                mid_y_r_pixel = (r_shoulder[1] + r_hip[1]) // 2
                alpha_front = (45.0 - pitch_deg) / 45.0
                guide_color = (
                    int(0 * alpha_front),
                    int(80 * alpha_front),
                    int(0 * alpha_front),
                )
                cv.line(
                    canvas,
                    (0, mid_y_r_pixel),
                    (self.width, mid_y_r_pixel),
                    guide_color,
                    1,
                    cv.LINE_AA,
                )

            if pitch_deg >= 45.0:
                alpha_top = (pitch_deg - 45.0) / 45.0
                guide_color = (
                    int(0 * alpha_top),
                    int(80 * alpha_top),
                    int(0 * alpha_top),
                )
                cv.line(
                    canvas,
                    (mid_shoulder_x - 70, mid_shoulder_y),
                    (mid_shoulder_x + 70, mid_shoulder_y),
                    guide_color,
                    1,
                    cv.LINE_AA,
                )

            # 胴体・骨格
            if pitch_deg < 60.0:
                body_poly = np.array(
                    [l_shoulder, r_shoulder, r_hip, l_hip], dtype=np.int32
                )
                overlay_body = canvas.copy()
                cv.fillPoly(overlay_body, [body_poly], self.COLOR_BODY_FILL, cv.LINE_AA)
                cv.polylines(
                    overlay_body, [body_poly], True, self.COLOR_BONE, 2, cv.LINE_AA
                )
                cv.addWeighted(overlay_body, 0.6, canvas, 0.4, 0, canvas)

            cv.circle(canvas, (nose[0], nose[1] - 5), 13, (60, 60, 60), -1, cv.LINE_AA)
            cv.circle(
                canvas, (nose[0], nose[1] - 5), 13, (120, 120, 120), 1, cv.LINE_AA
            )

            # 体幹（胴体）
            cv.line(canvas, l_shoulder, r_shoulder, self.COLOR_BONE, 6, cv.LINE_AA)
            cv.line(canvas, r_shoulder, r_hip, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, l_shoulder, l_hip, self.COLOR_BONE, 4, cv.LINE_AA)
            cv.line(canvas, r_hip, l_hip, self.COLOR_BONE, 5, cv.LINE_AA)

            # 腕の描画
            r_arm_color = self.COLOR_BONE
            l_arm_color = self.COLOR_BONE

            if is_height_ok and show_pop:
                if pop_wrist_idx == 16:
                    r_arm_color = self.COLOR_ACTIVE
                elif pop_wrist_idx == 15:
                    l_arm_color = self.COLOR_ACTIVE

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

            # --- 正面モード フィードバック表示 ---
            if pitch_deg < 45.0 and show_pop:
                target_wrist_px = get_px(pop_wrist_idx)
                txt, color = (
                    ("OK", self.COLOR_LIME_OK)
                    if is_height_ok
                    else ("NG", self.COLOR_RED_TEXT)
                )
                offset_x_txt = 12 if pop_wrist_idx == 15 else -35
                cv.putText(
                    canvas,
                    txt,
                    (target_wrist_px[0] + offset_x_txt, target_wrist_px[1] - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv.LINE_AA,
                )

            # --- 真上モード フィードバック表示 ---
            if pitch_deg >= 45.0:
                vec_x = (
                    world_landmarks[0].x
                    - (world_landmarks[11].x + world_landmarks[12].x) / 2.0
                )
                vec_z = (
                    world_landmarks[0].z
                    - (world_landmarks[11].z + world_landmarks[12].z) / 2.0
                )
                norm = np.sqrt(vec_x**2 + vec_z**2)

                if norm > 0:
                    vx = vec_x / norm
                    vz = vec_z / norm

                    start_pt = (int(nose[0]), int(nose[1] - 5))
                    end_pt = (
                        int(start_pt[0] + vx * 25),
                        int(start_pt[1] - vz * 25),
                    )

                    cv.arrowedLine(
                        canvas,
                        start_pt,
                        end_pt,
                        (0, 255, 0),
                        2,
                        tipLength=0.35,
                        line_type=cv.LINE_AA,
                    )

                    current_angle = np.degrees(np.arctan2(-vec_z, vec_x))
                    current_angle = np.clip(current_angle, 0.0, 180.0)

                    if current_angle > 90:
                        diff_deg = int(max(0, 130 - current_angle))
                    else:
                        diff_deg = int(max(0, current_angle - 30))

                    if diff_deg > 0:
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

                # 💡 【改修】該当する2項目のみをスコア表示
                if scores is not None:
                    score_ll = scores.get("LL_score", 0)
                    score_lr = scores.get("LR_score", 0)
                    score_pl = scores.get("PL_score", 0)
                    score_pr = scores.get("PR_score", 0)

                    items_to_draw = []

                    # 左側ピーク時：Look Left & Point Left のみ
                    if show_left_score_pop:
                        items_to_draw.append(
                            (f"Look Left: {score_ll} pts", (15, 45), score_ll >= 80)
                        )
                        items_to_draw.append(
                            (
                                f"Point Left: {score_pl} pts",
                                (15, self.split_y - 15),
                                score_pl >= 80,
                            )
                        )

                    # 右側ピーク時：Look Right & Point Right のみ
                    if show_right_score_pop:
                        items_to_draw.append(
                            (
                                f"Look Right: {score_lr} pts",
                                (self.width - 200, 45),
                                score_lr >= 80,
                            )
                        )
                        items_to_draw.append(
                            (
                                f"Point Right: {score_pr} pts",
                                (self.width - 200, self.split_y - 15),
                                score_pr >= 80,
                            )
                        )

                    for text, pos, is_ok in items_to_draw:
                        color = (0, 255, 0) if is_ok else (0, 0, 255)
                        (text_w, text_h), _ = cv.getTextSize(
                            text, cv.FONT_HERSHEY_SIMPLEX, 0.43, 1
                        )
                        box_x1, box_y1 = pos[0] - 6, pos[1] - text_h - 6
                        box_x2, box_y2 = pos[0] + text_w + 6, pos[1] + 4
                        cv.rectangle(
                            canvas, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1
                        )
                        cv.rectangle(
                            canvas, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 60), 1
                        )
                        cv.putText(
                            canvas,
                            text,
                            pos,
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            color,
                            1,
                            cv.LINE_AA,
                        )

        except Exception as e:
            pass

    def generate_animated_frame(
        self,
        world_landmarks,
        raw_camera_frame=None,
        pitch_deg=0.0,
        is_height_ok=False,
        scores=None,
        show_pop=False,
        pop_wrist_idx=16,
        show_left_score_pop=False,  # 💡 フラグ追加
        show_right_score_pop=False,  # 💡 フラグ追加
    ):
        canvas = np.full((self.height, self.width, 3), self.COLOR_BG, dtype=np.uint8)

        for x in range(0, self.width, 50):
            cv.line(canvas, (x, 0), (x, self.split_y), self.COLOR_GRID, 1)
        for y in range(0, self.split_y, 50):
            cv.line(canvas, (0, y), (self.width, y), self.COLOR_GRID, 1)

        self.draw_3d_skeleton(
            canvas,
            world_landmarks,
            pitch_deg,
            is_height_ok,
            scores,
            show_pop,
            pop_wrist_idx,
            show_left_score_pop,
            show_right_score_pop,
        )

        if raw_camera_frame is not None:
            resized_cam = cv.resize(raw_camera_frame, (self.width, self.split_y))
            canvas[self.split_y : self.height, 0 : self.width] = resized_cam

        cv.line(canvas, (0, self.split_y), (self.width, self.split_y), (80, 80, 80), 2)

        mode_title = (
            "3D POSE ANIMATION (HIGH VIEW)"
            if pitch_deg < 45.0
            else "3D POSE ANIMATION (TOP VIEW)"
        )
        cv.putText(
            canvas,
            mode_title,
            (16, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            mode_title,
            (15, 21),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_TEXT,
            1,
            cv.LINE_AA,
        )

        cv.putText(
            canvas,
            "ACTUAL CAMERA VIEW",
            (16, self.split_y + 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            "ACTUAL CAMERA VIEW",
            (15, self.split_y + 21),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_TEXT,
            1,
            cv.LINE_AA,
        )

        return canvas
