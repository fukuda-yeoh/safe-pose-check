import cv2
import numpy as np
from funcs import rotate_marker_corners

# Arucoの辞書を指定
dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

cameraMatrix = np.array(
    [
        [652.0380903058872, 0.0, 327.60194156388894],
        [0.0, 654.1462516033431, 237.0926726679871],
        [0.0, 0.0, 1.0],
    ]
)
distCoeffs = np.array(
    [
        -0.002680106355429372,
        -0.0025897706030070912,
        -0.0002925868049691957,
        0.0022478611989058,
        0.6157793950053083,
    ]
)
# Webカメラを起動
cap = cv2.VideoCapture(1)
# 入力動画ファイルのパス
# video_path = r"C:\Users\ryono\SafePoseCheck\sample_data\Aruco.mp4"
# cap = cv2.VideoCapture(video_path)

# 表示サイズを設定
display_width = 2160
display_height = 3840

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        break

    # Arucoマーカーの検出
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dict_aruco)

    # 検出されたマーカーがあれば処理
    if ids is not None:
        # マーカーの枠を表示
        cv2.aruco.drawDetectedMarkers(frame, corners)
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.4, cameraMatrix, distCoeffs
        )

        markercorners, mrv = rotate_marker_corners(rvecs[0], 0.4, tvecs[0])
        print(markercorners[0])
        print(markercorners[1])
        print(markercorners[1] - markercorners[0])

        for i in range(ids.size):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)

    # フレームをリサイズ
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # リサイズしたフレームを表示
    cv2.imshow("Aruco Marker Horizontal Vector", resized_frame)

    # 'q' キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 終了処理
cap.release()
