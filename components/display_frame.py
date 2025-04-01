import cv2 as cv
import numpy as np


class DisplayFrame:
    def __init__(self):
        self.base_frame = self._generate_base_frame()
        self.dynamic_frames = {
            "Camera": [None, (20, 34, 768, 432)],
            "Look Left": [None, (810, 100, 100, 100)],
            "Look Right": [None, (970, 100, 100, 100)],
            "Point Left": [None, (810, 300, 100, 100)],
            "Point Right": [None, (970, 300, 100, 100)],
        }
        self.display_frame = self.base_frame

    def set_dynamic_img(self, key, frame):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = frame

    def clear_dynamic_img(self, key):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = None

    def clear_all_dynamic_img(self):
        for key in self.dynamic_frames.keys():
            self.dynamic_frames[key][0] = None

    def update_display(self):
        self.display_frame = self.base_frame.copy()
        for img, params in self.dynamic_frames.values():
            if img is not None:
                self._paste(self.display_frame, img, *params)
        return self.display_frame

    def _generate_base_frame(self):
        frame = np.zeros((500, 1100, 3), dtype=np.uint8)
        self._draw_heading_text(frame, "Look Left", (810, 90))
        self._draw_heading_text(frame, "Look Right", (970, 90))
        self._draw_heading_text(frame, "Point Left", (810, 290))
        self._draw_heading_text(frame, "Point Right", (970, 290))
        return frame

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
        roi_dst[:] = roi_resized


if __name__ == "__main__":
    display = DisplayFrame()
    white_img = np.ones((10, 10, 3)) * 255
    blue_img = np.zeros((10, 10, 3))
    blue_img[:, :, 0] = 255

    display.set_dynamic_img("Camera", white_img)
    display.set_dynamic_img("Look Left", blue_img)
    display.set_dynamic_img("Look Right", blue_img)
    display.set_dynamic_img("Point Left", blue_img)
    display.set_dynamic_img("Point Right", blue_img)

    display.update_display()
    cv.imshow("display", display.display_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
