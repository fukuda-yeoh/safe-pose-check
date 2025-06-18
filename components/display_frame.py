import cv2 as cv
import numpy as np


class DisplayFrame:
    def __init__(self):
        self.dynamic_frames = {
            "Camera": [None, (0, 0, 0, 0)],
            # Image1とImage2は汎用的な枠として残しつつ、
            # 今回はLeft/Rightcheck画像を専用で管理
            "LeftCheckImage": [None, (0, 0, 0, 0)],  # 表示位置は動的に設定
            "RightCheckImage": [None, (0, 0, 0, 0)],  # 表示位置は動的に設定
        }
        self.display_frame = None

        self.traffic_light_states = {
            "Look Left": False,
            "Look Right": False,
            "Point Left": False,
            "Point Right": False,
        }
        self.light_radius = 20
        self.light_margin = 30
        self.light_spacing = 80

        self.person_bbox = None
        self.dim_mask_needed = False

        self.confirmed_count = 0
        self.entered_count = 0

        # 新しい画像プロパティ
        self.left_check_img = None
        self.right_check_img = None

    def set_dynamic_img(self, key, frame, pos_params=None):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = frame
            if pos_params:
                self.dynamic_frames[key][1] = pos_params

    def clear_dynamic_img(self, key):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = None

    def clear_all_dynamic_img(self):
        # dynamic_framesに登録されているものをすべてクリア
        for key in self.dynamic_frames.keys():
            self.dynamic_frames[key][0] = None
        self.reset_traffic_lights()
        self.person_bbox = None
        self.dim_mask_needed = False
        # チェック画像もクリア
        self.left_check_img = None
        self.right_check_img = None

    def set_traffic_light_state(self, event_name, is_detected):
        if event_name in self.traffic_light_states:
            self.traffic_light_states[event_name] = is_detected

    def reset_traffic_lights(self):
        for key in self.traffic_light_states:
            self.traffic_light_states[key] = False

    def set_person_detection_info(self, bbox, dim_mask):
        self.person_bbox = bbox
        self.dim_mask_needed = dim_mask

    def set_counts(self, confirmed: int, entered: int):
        self.confirmed_count = confirmed
        self.entered_count = entered

    def set_check_image(
        self, direction: str, image_data: np.ndarray = None, pos_params: tuple = None
    ):
        """
        指定された方向のチェック画像を設定します。
        :param direction: 'Left' または 'Right'
        :param image_data: 表示する画像データ (Noneでクリア)
        :param pos_params: (x, y, w, h) のタプルで、画像の表示位置とサイズ
        """
        if direction == "Left":
            self.left_check_img = image_data
            if pos_params:
                self.dynamic_frames["LeftCheckImage"][1] = pos_params
        elif direction == "Right":
            self.right_check_img = image_data
            if pos_params:
                self.dynamic_frames["RightCheckImage"][1] = pos_params

    def update_display(self):
        camera_frame_data = self.dynamic_frames.get("Camera")
        if camera_frame_data and camera_frame_data[0] is not None:
            camera_frame = camera_frame_data[0]
            self.display_frame = camera_frame.copy()

            # 薄暗い効果を適用
            if self.dim_mask_needed and self.person_bbox:
                self._apply_dim_effect(self.display_frame, self.person_bbox)

            # 人物の枠を描画
            if self.person_bbox:
                self._draw_person_bbox(self.display_frame, self.person_bbox)

            # LeftCheckImageの貼り付け
            if self.left_check_img is not None:
                _, params = self.dynamic_frames["LeftCheckImage"]
                self._paste(self.display_frame, self.left_check_img, *params)

            # RightCheckImageの貼り付け
            if self.right_check_img is not None:
                _, params = self.dynamic_frames["RightCheckImage"]
                self._paste(self.display_frame, self.right_check_img, *params)

            # カウント表示の描画 (既存のロジック)
            if self.display_frame is not None:  # カメラフレームがNoneでないことを再確認
                frame_h, frame_w, _ = self.display_frame.shape

                count_text = (
                    f"Confirmed {self.confirmed_count} / Entered {self.entered_count}"
                )
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)
                background_color = (0, 0, 0)

                (text_width, text_height), baseline = cv.getTextSize(
                    count_text, font, font_scale, font_thickness
                )

                padding = 10
                bg_x1 = (frame_w - text_width) // 2 - padding
                bg_y1 = padding
                bg_x2 = (frame_w + text_width) // 2 + padding
                bg_y2 = padding + text_height + baseline + padding

                cv.rectangle(
                    self.display_frame,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    background_color,
                    -1,
                )

                text_x = (frame_w - text_width) // 2
                text_y = padding + text_height + baseline // 2
                cv.putText(
                    self.display_frame,
                    count_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv.LINE_AA,
                )

                # 信号機の描画 (既存のロジック)
                light_y = frame_h - self.light_margin - self.light_radius

                light_names = ["Look Left", "Look Right", "Point Left", "Point Right"]
                total_lights_width = (
                    len(light_names) * (self.light_radius * 2 + self.light_spacing)
                ) - self.light_spacing
                start_x = (frame_w - total_lights_width) // 2 + self.light_radius

                for i, name in enumerate(light_names):
                    light_color = (
                        (0, 0, 255)
                        if not self.traffic_light_states[name]
                        else (0, 255, 0)
                    )
                    light_x = start_x + i * (self.light_radius * 2 + self.light_spacing)
                    self._draw_traffic_light(
                        self.display_frame,
                        (light_x, light_y),
                        self.light_radius,
                        light_color,
                    )
                    text_x = light_x - (
                        cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] // 2
                    )
                    cv.putText(
                        self.display_frame,
                        name,
                        (text_x, light_y + self.light_radius + 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv.LINE_AA,
                    )
        else:
            self.display_frame = None

        return self.display_frame

    @staticmethod
    def _draw_traffic_light(frame, center_coords, radius, color):
        cv.circle(frame, center_coords, radius, color, -1)
        cv.circle(frame, center_coords, radius, (255, 255, 255), 2)

    @staticmethod
    def _draw_person_bbox(frame, bbox, color=(0, 255, 255), thickness=3):
        x1, y1, x2, y2 = bbox
        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def _apply_dim_effect(frame, bbox, dim_factor=0.6):
        overlay = frame.copy()
        output = frame.copy()

        mask = np.zeros(frame.shape, dtype=np.uint8)
        x1, y1, x2, y2 = bbox

        cv.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        mask_inv = cv.bitwise_not(mask)

        dim_area = cv.bitwise_and(overlay, mask_inv)
        dim_area = cv.convertScaleAbs(dim_area * dim_factor)

        person_area = cv.bitwise_and(overlay, mask)

        cv.add(dim_area, person_area, output)

        frame[:] = output[:]

    @staticmethod
    def _draw_heading_text(
        frame,
        text,
        pos,
        font=cv.FONT_HERSHEY_SIMPLEX,
        font_scale=0.6,
        color=(255, 255, 255),
        thickness=2,
    ):
        cv.putText(
            frame,
            text,
            pos,
            font,
            font_scale,
            color,
            thickness,
        )

    @staticmethod
    def _paste(dst, src, x, y, a, b):
        # srcがNoneの場合は何もしない
        if src is None:
            return

        resized_img = cv.resize(src, (a, b))

        # 貼り付け先の範囲をチェックし、はみ出す部分を調整
        if x >= dst.shape[1] or y >= dst.shape[0]:
            return

        # 貼り付け元の画像が3チャンネルでない場合、3チャンネルに変換 (PNGのアルファチャンネル対応)
        if resized_img.shape[2] == 4:  # RGBAの場合
            # アルファチャンネルをマスクとして使用
            alpha_s = resized_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                dst[y : y + b, x : x + a, c] = (
                    alpha_s * resized_img[:, :, c]
                    + alpha_l * dst[y : y + b, x : x + a, c]
                )
            return  # 貼り付け処理が完了したので終了
        elif resized_img.shape[2] == 1:  # グレースケールの場合
            resized_img = cv.cvtColor(resized_img, cv.COLOR_GRAY2BGR)

        w = (
            min(dst.shape[1] - x, resized_img.shape[1])
            if x >= 0
            else min(max(resized_img.shape[1] + x, 0), dst.shape[1])
        )
        h = (
            min(dst.shape[0] - y, resized_img.shape[0])
            if y >= 0
            else min(max(resized_img.shape[0] + y, 0), dst.shape[0])
        )
        u = 0 if x >= 0 else min(-x, resized_img.shape[1] - 1)
        v = 0 if y >= 0 else min(-y, resized_img.shape[0] - 1)
        px = max(x, 0)
        py = max(y, 0)

        roi_dst = dst[py : py + h, px : px + w]
        roi_resized = resized_img[v : v + h, u : u + w]

        if (
            roi_dst.shape[0] != roi_resized.shape[0]
            or roi_dst.shape[1] != roi_resized.shape[1]
        ):
            roi_resized = cv.resize(roi_resized, (roi_dst.shape[1], roi_dst.shape[0]))

        roi_dst[:] = roi_resized


if __name__ == "__main__":
    display = DisplayFrame()
    white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_img[:, :, 0] = 255

    display.set_dynamic_img("Camera", white_img, (0, 0, 640, 480))

    display.set_traffic_light_state("Look Left", True)
    display.set_traffic_light_state("Point Right", True)

    test_bbox = (100, 100, 500, 400)
    display.set_person_detection_info(test_bbox, True)

    display.set_counts(confirmed=5, entered=12)

    # テスト用に画像を読み込み設定
    test_left_img = cv.imread("Leftcheck.png", cv.IMREAD_UNCHANGED)  # アルファチャンネルも読み込む
    test_right_img = cv.imread("Rightcheck.png", cv.IMREAD_UNCHANGED)  # アルファチャンネルも読み込む

    # 画像のサイズと位置を設定
    img_width, img_height = 150, 150  # 表示サイズ
    cam_w, cam_h = 640, 480  # カメラ画像の幅と高さ（テスト用）

    # 左側の画像の位置（例: カメラ映像の左上）
    left_img_x = 10
    left_img_y = 10
    display.set_check_image(
        "Left", test_left_img, (left_img_x, left_img_y, img_width, img_height)
    )

    # 右側の画像の位置（例: カメラ映像の右上）
    right_img_x = cam_w - img_width - 10
    right_img_y = 10
    display.set_check_image(
        "Right", test_right_img, (right_img_x, right_img_y, img_width, img_height)
    )

    display.update_display()
    if display.display_frame is not None:
        cv.imshow("display", display.display_frame)
        cv.waitKey(0)
    cv.destroyAllWindows()
