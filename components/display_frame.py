import cv2 as cv
import numpy as np


class DisplayFrame:
    def __init__(self):
        self.dynamic_frames = {
            "Camera": [None, (0, 0, 0, 0)],
            "Image1": [None, (10, 10, 200, 200)],
            "Image2": [None, (220, 10, 200, 200)],
        }
        self.display_frame = None

        # 信号機の状態を保持する辞書を追加
        # キーはイベント名に対応し、値はTrue(青)またはFalse(赤)
        self.traffic_light_states = {
            "Look Left": False,
            "Look Right": False,
            "Point Left": False,
            "Point Right": False,
        }
        # 信号機の描画設定
        self.light_radius = 20
        self.light_margin = 30
        self.light_spacing = 80  # 信号機間のスペース

    def set_dynamic_img(self, key, frame, pos_params=None):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = frame
            if pos_params:
                self.dynamic_frames[key][1] = pos_params

    def clear_dynamic_img(self, key):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = None

    def clear_all_dynamic_img(self):
        for key in self.dynamic_frames.keys():
            self.dynamic_frames[key][0] = None
        # 全ての動的画像をクリアする際に信号機の状態もリセット
        self.reset_traffic_lights()

    def set_traffic_light_state(self, event_name, is_detected):
        """
        指定されたイベント名の信号機の状態を更新します。
        is_detectedがTrueなら青、Falseなら赤になります。
        """
        if event_name in self.traffic_light_states:
            self.traffic_light_states[event_name] = is_detected

    def reset_traffic_lights(self):
        """全ての信号機の状態を赤にリセットします。"""
        for key in self.traffic_light_states:
            self.traffic_light_states[key] = False

    def update_display(self):
        camera_frame_data = self.dynamic_frames.get("Camera")
        if camera_frame_data and camera_frame_data[0] is not None:
            camera_frame = camera_frame_data[0]
            self.display_frame = camera_frame.copy()

            # カメラ映像以外の動的画像を貼り付け（これは変更なし）
            for key, (img, params) in self.dynamic_frames.items():
                if key != "Camera" and img is not None:
                    self._paste(self.display_frame, img, *params)

            # ここから信号機の描画を追加
            if self.display_frame is not None:
                frame_h, frame_w, _ = self.display_frame.shape
                # 信号機を描画するY座標（カメラ映像の下部）
                light_y = frame_h - self.light_margin - self.light_radius

                # 各信号機を描画
                light_names = ["Look Left", "Look Right", "Point Left", "Point Right"]
                # 画面中央に配置するために、全体幅を計算
                total_lights_width = (
                    len(light_names) * (self.light_radius * 2 + self.light_spacing)
                ) - self.light_spacing
                start_x = (frame_w - total_lights_width) // 2 + self.light_radius

                for i, name in enumerate(light_names):
                    light_color = (
                        (0, 0, 255)
                        if not self.traffic_light_states[name]
                        else (0, 255, 0)
                    )  # 赤(False)または緑(True)
                    light_x = start_x + i * (self.light_radius * 2 + self.light_spacing)
                    self._draw_traffic_light(
                        self.display_frame,
                        (light_x, light_y),
                        self.light_radius,
                        light_color,
                    )
                    # 信号機の下にテキストを描画
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
        """信号機として円を描画します。"""
        cv.circle(frame, center_coords, radius, color, -1)  # -1 で塗りつぶし
        cv.circle(frame, center_coords, radius, (255, 255, 255), 2)  # 白い境界線

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
        resized_img = cv.resize(src, (a, b))

        if x >= dst.shape[1] or y >= dst.shape[0]:
            return

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
    # テストのためにImage1とImage2のセットを削除
    # display.set_dynamic_img("Image1", blue_img)
    # display.set_dynamic_img("Image2", blue_img)

    display.set_traffic_light_state("Look Left", True)  # 例としてLook Leftを青に
    display.set_traffic_light_state("Point Right", True)  # 例としてPoint Rightを青に

    display.update_display()
    if display.display_frame is not None:
        cv.imshow("display", display.display_frame)
        cv.waitKey(0)
    cv.destroyAllWindows()
