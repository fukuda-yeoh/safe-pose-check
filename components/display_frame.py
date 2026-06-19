import cv2 as cv
import numpy as np


class DisplayFrame:
    def __init__(self):
        self.display_frame = None
        self.dynamic_imgs = {}
        self.text_feedbacks = []
        self.result_img = None
        self.result_img_rect = None

        self.traffic_lights = {
            "Look Left": False,
            "Look Right": False,
            "Point Left": False,
            "Point Right": False,
        }
        self.confirmed_count = 0
        self.entered_count = 0
        self.is_person_detected = False
        self.person_bbox_coords = None

    def set_dynamic_img(self, name, img, rect):
        self.dynamic_imgs[name] = (img, rect)

    def clear_dynamic_img(self, name):
        if name in self.dynamic_imgs:
            del self.dynamic_imgs[name]

    def set_traffic_light_state(self, name, state):
        if name in self.traffic_lights:
            self.traffic_lights[name] = state

    def reset_traffic_lights(self):
        for name in self.traffic_lights:
            self.traffic_lights[name] = False

    def set_counts(self, confirmed_count, entered_count):
        self.confirmed_count = confirmed_count
        self.entered_count = entered_count

    def set_person_detection_info(self, bbox_coords, is_detected):
        self.person_bbox_coords = bbox_coords
        self.is_person_detected = is_detected

    def add_text_feedback(
        self, text, pos, color=(0, 255, 0), font_scale=0.6, thickness=2
    ):
        self.text_feedbacks.append((text, pos, color, font_scale, thickness))

    def clear_text_feedbacks(self):
        self.text_feedbacks.clear()

    def set_result_image(self, img, rect=None):
        self.result_img = img
        self.result_img_rect = rect

    def update_display(self):
        # dynamic_imgs からベース画像を取得
        camera_data = self.dynamic_imgs.get("Camera")
        if camera_data is None:
            self.display_frame = None
            return

        base_img, (x, y, w, h) = camera_data
        self.display_frame = base_img.copy()

        # 1. テキストフィードバック（案内テキストなど）の描画
        for text, pos, color, font_scale, thickness in self.text_feedbacks:
            cv.putText(
                self.display_frame,
                text,
                pos,
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv.LINE_AA,
            )

        # 2. 判定スタンプ（OK/NG/案内画像）の重ね合わせ処理（アルファチャンネル透過対応）
        if self.result_img is not None:
            r_img = self.result_img
            if self.result_img_rect is not None:
                rx, ry, rw, rh = self.result_img_rect
                r_img = cv.resize(r_img, (rw, rh))
            else:
                ry, rx, _ = r_img.shape
                rw, rh = f_w, f_h

            if r_img.shape[2] == 4:  # アルファチャンネル透過がある場合
                alpha_channel = r_img[:, :, 3] / 255.0
                overlay_img = r_img[:, :, :3]
                for c in range(3):
                    self.display_frame[ry : ry + rh, rx : rx + rw, c] = (
                        alpha_channel * overlay_img[:, :, c]
                        + (1.0 - alpha_channel)
                        * self.display_frame[ry : ry + rh, rx : rx + rw, c]
                    )
            else:
                self.display_frame[ry : ry + rh, rx : rx + rw] = r_img

        # ======================================================================
        # 【修正】下部信号機（Traffic Lights）および左上カウンタ（PASS/FAIL）の描画コードを完全削除
        # ======================================================================
        # (以前ここに書かれていた cv.circle や cv.putText の塊を完全に排除しました)
        # ======================================================================

    def draw_score_board(self, scores: dict):
        """Monitor Windowでは使用しないため無効化"""
        pass
