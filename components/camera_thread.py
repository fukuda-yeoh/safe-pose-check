import time
from queue import Queue
from threading import Thread
import numpy as np
import cv2 as cv


class CameraThread(Thread):
    def __init__(self, camera_no, cam_matrix, dist_coeffs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_no = camera_no
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.queue = Queue(maxsize=30)

        self.stop_flag = False
        self.cap = None
        self.frame = None
        self.start_time = time.time()

    def run(self):
        self.cap = cv.VideoCapture(self.camera_no)
        # カメラが開けない場合はエラーメッセージを表示して終了
        if not self.cap.isOpened():
            print(f"エラー: カメラ {self.camera_no} を開けませんでした。")
            self.stop_flag = True  # スレッドを停止させる
            return

        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                # フレームの読み込みに失敗した場合の処理
                print("警告: カメラからのフレーム読み込みに失敗しました。")
                time.sleep(0.1)  # 少し待ってからリトライ
                continue

            timestamp = int((time.time() - self.start_time) * 1000)

            # 修正: frame.copy() を使用して、入力と出力のメモリ空間を分ける
            undistorted_frame = cv.undistort(
                frame.copy(), self.cam_matrix, self.dist_coeffs
            )  # 歪み補正

            self.queue.put((timestamp, undistorted_frame))
        self.cap.release()

    def stop(self):
        self.stop_flag = True


if __name__ == "__main__":
    # cam_matrix と dist_coeffs がないため、テストコードは以前と同様に動作しない
    # ダミーの行列と係数を用意するか、main_live.py のように適切な値を渡す必要がある
    # 例としてダミーの値を設定 (実際のキャリブレーション値を使用してください)
    dummy_camera_matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    dummy_dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # 歪みがないと仮定

    cam_no = 0  # デフォルトカメラ
    camera_thread = CameraThread(cam_no, dummy_camera_matrix, dummy_dist_coeffs)
    camera_thread.start()

    while True:
        try:
            t, raw_frame = camera_thread.queue.get(timeout=1)  # タイムアウトを設定
        except Exception:
            # キューが空の場合など
            continue

        if raw_frame is not None:
            cv.imshow(f"Camera {cam_no}", raw_frame)
            if cv.waitKey(1) == ord("q"):
                camera_thread.stop()
                break
        else:
            print("Received None frame from camera queue.")

    camera_thread.stop()
    print("Camera thread stopped.")
    cv.destroyAllWindows()
