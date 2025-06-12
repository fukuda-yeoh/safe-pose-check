import cv2 as cv
import numpy as np


class DisplayFrame:
    def __init__(self):
        self.dynamic_frames = {
            "Camera": [None, (0, 0, 0, 0)],
            "Image1": [None, (10, 10, 200, 200)],
            "Image2": [None, (220, 10, 200, 200)],
            "Image3": [None, (10, 220, 200, 200)],
            "Image4": [None, (220, 220, 200, 200)],
        }
        self.display_frame = None
        self.status_text = None
        self.status_text_pos = (0, 0)
        # self.counter_text = None # 削除
        # self.counter_text_pos = (0, 0) # 削除

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
        self.clear_status_text()
        # self.clear_counter_text() # 削除

    def set_status_text(self, text, pos):
        self.status_text = text
        self.status_text_pos = pos

    def clear_status_text(self):
        self.status_text = None
        self.status_text_pos = (0, 0)

    # def set_counter_text(self, text, pos): # 削除
    #     self.counter_text = text
    #     self.counter_text_pos = pos

    # def clear_counter_text(self): # 削除
    #     self.counter_text = None
    #     self.counter_text_pos = (0, 0)

    def update_display(self):
        camera_frame_data = self.dynamic_frames.get("Camera")
        if camera_frame_data and camera_frame_data[0] is not None:
            camera_frame = camera_frame_data[0]
            self.display_frame = camera_frame.copy()

            for key, (img, params) in self.dynamic_frames.items():
                if key != "Camera" and img is not None:
                    self._paste(self.display_frame, img, *params)

            # OK/NGテキストを描画
            if self.status_text is not None:
                color = (0, 255, 0) if self.status_text == "OK" else (0, 0, 255)
                DisplayFrame._draw_text_with_background(
                    self.display_frame,
                    self.status_text,
                    self.status_text_pos,
                    font_scale=1.5,
                    thickness=3,
                    color=color,
                    bg_color=(0, 0, 0),
                    center_x_aligned=True,
                )

            # カウンタテキストを描画 (main_live.pyで直接描画するため、ここから削除)
            # if self.counter_text is not None:
            #     DisplayFrame._draw_text_with_background(
            #         self.display_frame,
            #         self.counter_text,
            #         self.counter_text_pos,
            #         font_scale=1.0,
            #         thickness=2,
            #         color=(255, 255, 255),
            #         center_x_aligned=False
            #     )
        else:
            self.display_frame = None

        return self.display_frame

    @staticmethod
    def _draw_text_with_background(
        frame,
        text,
        pos,
        font=cv.FONT_HERSHEY_SIMPLEX,
        font_scale=0.6,
        color=(255, 255, 255),
        thickness=2,
        bg_color=(0, 0, 0),
        center_x_aligned=False,
    ):
        (text_width, text_height), baseline = cv.getTextSize(
            text, font, font_scale, thickness
        )
        x, y = pos

        if center_x_aligned:
            x = max(0, x - text_width // 2)

        cv.rectangle(
            frame,
            (x, y - text_height - baseline),
            (x + text_width, y + baseline),
            bg_color,
            -1,
        )

        cv.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
        )

    @staticmethod
    def _paste(dst, src, x, y, a, b):
        resized_img = cv.resize(src, (a, b))

        h_dst, w_dst, _ = dst.shape
        h_src, w_src, _ = resized_img.shape

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_dst, x + w_src)
        y2 = min(h_dst, y + h_src)

        u1 = 0
        v1 = 0
        if x < 0:
            u1 = -x
        if y < 0:
            v1 = -y
        u2 = u1 + (x2 - x1)
        v2 = v1 + (y2 - y1)

        if x1 < x2 and y1 < y2:
            dst[y1:y2, x1:x2] = resized_img[v1:v2, u1:u2]


if __name__ == "__main__":
    display = DisplayFrame()
    white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    display.set_dynamic_img("Camera", white_img, (0, 0, 640, 480))
    display.set_status_text("NG", (320, 100))

    display.update_display()
    if display.display_frame is not None:
        cv.imshow("display", display.display_frame)
        cv.waitKey(0)
    cv.destroyAllWindows()
