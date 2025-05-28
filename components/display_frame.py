import cv2 as cv
import numpy as np


class DisplayFrame:
    def __init__(self):
        # ベースフレームの生成は不要になります
        # self.base_frame = self._generate_base_frame()
        self.dynamic_frames = {
            "Camera": [None, (0, 0, 0, 0)],  # カメラ映像のサイズと位置は動的に設定するため、初期値は0
            "Image1": [None, (10, 10, 200, 200)],  # カメラ映像の左上に表示する例
            "Image2": [None, (220, 10, 200, 200)],  # カメラ映像の右上に表示する例
        }
        self.display_frame = None  # 初期表示フレームはNoneに

    def set_dynamic_img(self, key, frame, pos_params=None):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = frame
            if pos_params:  # 位置とサイズが指定された場合のみ更新
                self.dynamic_frames[key][1] = pos_params

    def clear_dynamic_img(self, key):
        if key in self.dynamic_frames:
            self.dynamic_frames[key][0] = None

    def clear_all_dynamic_img(self):
        for key in self.dynamic_frames.keys():
            self.dynamic_frames[key][0] = None

    def update_display(self):
        # 'Camera' キーの画像がNoneでないことを確認
        camera_frame_data = self.dynamic_frames.get("Camera")
        if camera_frame_data and camera_frame_data[0] is not None:
            camera_frame = camera_frame_data[0]
            # カメラ映像をdisplay_frameのベースとして使用
            self.display_frame = camera_frame.copy()

            # カメラ映像以外の動的画像を貼り付け
            for key, (img, params) in self.dynamic_frames.items():
                if key != "Camera" and img is not None:  # カメラ映像以外を貼り付け
                    self._paste(self.display_frame, img, *params)
        else:
            # カメラ映像がない場合は、黒い画面を表示するか、何もしないか選択
            # 今回は、カメラ映像がない場合はNoneのままにする
            self.display_frame = None

        return self.display_frame

    # _generate_base_frame メソッドは不要になります
    # def _generate_base_frame(self):
    #     frame = np.zeros((3840, 2160, 3), dtype=np.uint8)
    #     return frame

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
        # 指定されたサイズにリサイズ
        resized_img = cv.resize(src, (a, b))

        # 貼り付け先の範囲をチェックし、はみ出す部分を調整
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

        # 貼り付け
        # resized_img のサイズと貼り付け先のROIのサイズが一致しない可能性があるため、
        # roi_resized の幅と高さを調整して貼り付けます。
        roi_dst = dst[py : py + h, px : px + w]
        roi_resized = resized_img[v : v + h, u : u + w]

        # サイズが異なる場合は、貼り付け先のROIのサイズに合わせて resized_img を再度リサイズ
        if (
            roi_dst.shape[0] != roi_resized.shape[0]
            or roi_dst.shape[1] != roi_resized.shape[1]
        ):
            roi_resized = cv.resize(roi_resized, (roi_dst.shape[1], roi_dst.shape[0]))

        roi_dst[:] = roi_resized


if __name__ == "__main__":
    display = DisplayFrame()
    # テスト用に大きな白い画像と青い画像を準備
    white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # カメラ映像の代わり
    blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_img[:, :, 0] = 255

    display.set_dynamic_img("Camera", white_img, (0, 0, 640, 480))  # カメラ映像のサイズと位置を設定
    display.set_dynamic_img("Image1", blue_img)
    display.set_dynamic_img("Image2", blue_img)

    display.update_display()
    if display.display_frame is not None:
        cv.imshow("display", display.display_frame)
        cv.waitKey(0)
    cv.destroyAllWindows()
