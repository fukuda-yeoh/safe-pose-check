import cv2
import numpy as np
import json
from defs import project_root

# 設定ファイルのパス
CONFIG_DIR = project_root() / "config"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "roi_config.json"

# グローバル変数
drawing_points = []
current_area_name = "roi_points"  # 現在設定中のエリア名
areas_to_define = {
    "roi_points": [],
    "display_check_area_points": [],
    "clear_roi_points": [],
}
area_color = {
    "roi_points": (0, 255, 0),
    "display_check_area_points": (255, 0, 255),
    "clear_roi_points": (0, 0, 255),
}
camera_matrix = None  # 仮の値を設定するか、キャリブレーションファイルから読み込む
dist_coeffs = None  # 仮の値を設定するか、キャリブレーションファイルから読み込む


# マウスイベントコールバック関数
def mouse_callback(event, x, y, flags, param):
    global drawing_points, current_area_name, areas_to_define

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左クリックで点を追加
        drawing_points.append((x, y))
        print(f"Added point for {current_area_name}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右クリックで最後の点を削除
        if drawing_points:
            drawing_points.pop()
            print(f"Removed last point from {current_area_name}")
    elif event == cv2.EVENT_MBUTTONDOWN:  # マウスの真ん中ボタンでリセット
        drawing_points = []
        print(f"Reset points for {current_area_name}")


def save_roi_config(frame_width, frame_height):
    global areas_to_define
    normalized_areas = {}
    for area_name, points in areas_to_define.items():
        normalized_points = []
        for x, y in points:
            normalized_points.append([x / frame_width, y / frame_height])
        normalized_areas[area_name] = normalized_points

    with open(CONFIG_FILE, "w") as f:
        json.dump(normalized_areas, f, indent=4)
    print(f"ROI configuration saved to {CONFIG_FILE}")


def load_roi_config():
    global areas_to_define
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            loaded_config = json.load(f)
            # ロードした設定を現在のareas_to_defineにマージ
            for area_name, normalized_points in loaded_config.items():
                if area_name in areas_to_define:
                    areas_to_define[area_name] = normalized_points
            print(f"ROI configuration loaded from {CONFIG_FILE}")
    else:
        print("No existing ROI configuration found. Starting fresh.")


def main():
    global drawing_points, current_area_name, areas_to_define, camera_matrix, dist_coeffs

    # カメラキャリブレーションデータ
    camera_matrix = np.array(
        [
            [635.8283543606941, 0.0, 253.67538054709416],
            [0.0, 636.5077974749544, 320.7767679600789],
            [0.0, 0.0, 1.0],
        ]
    )
    dist_coeffs = np.array(
        [
            -0.013855030664346785,
            0.036480883659782824,
            -0.004879021042728567,
            0.0019079058483680378,
            -0.21054703261551563,
        ],
        dtype=np.float32,
    )

    cap = cv2.VideoCapture(1)  # カメラ番号は適宜変更してください
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("ROI Setup", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Setup", mouse_callback)

    print("--- ROI Setup Mode ---")
    print("Click points counter-clockwise to define an area.")
    print("Right-click to remove the last point.")
    print("Middle-click to clear all points for the current area.")
    print("Press '1': Set ROI (current)")
    print("Press '2': Set Display Check Area")
    print("Press '3': Set Clear ROI Area")
    print("Press 's': Save current configuration")
    print("Press 'q': Quit (without saving current changes)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 歪み補正 (カメラキャリブレーションデータがあれば適用)
        if camera_matrix is not None and dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
            # ROI領域を切り取る場合（歪み補正後に黒い部分ができる場合）
            # x, y, w, h = roi
            # frame = frame[y:y+h, x:x+w]

        display_frame = frame.copy()

        frame_height, frame_width, _ = display_frame.shape

        # 現在設定中のエリアの点と線を描画
        if drawing_points:
            for i, point in enumerate(drawing_points):
                cv2.circle(display_frame, point, 5, (255, 0, 0), -1)  # 緑色の点
                if i > 0:
                    cv2.line(
                        display_frame, drawing_points[i - 1], point, (255, 0, 0), 2
                    )
            # 最後の点と最初の点を結ぶ線
            if len(drawing_points) > 2:
                cv2.line(
                    display_frame, drawing_points[-1], drawing_points[0], (255, 0, 0), 2
                )

        for area_name, points in areas_to_define.items():
            if points:
                for i, point in enumerate(points):
                    cv2.circle(
                        display_frame, point, 5, area_color[area_name], -1
                    )  # 緑色の点
                    if i > 0:
                        cv2.line(
                            display_frame,
                            points[i - 1],
                            point,
                            area_color[area_name],
                            2,
                        )
                # 最後の点と最初の点を結ぶ線
                if len(points) > 2:
                    cv2.line(
                        display_frame, points[-1], points[0], area_color[area_name], 2
                    )

        # 既に保存されている他のエリアを描画（もしあれば）
        # 正規化された座標をピクセル座標に戻して描画
        for area_name, normalized_points in areas_to_define.items():
            if area_name == current_area_name:
                continue  # 現在設定中のエリアはdrawing_pointsで描画するためスキップ

            if normalized_points:
                pixel_points = []
                for nx, ny in normalized_points:
                    pixel_points.append((int(nx * frame_width), int(ny * frame_height)))

                if len(pixel_points) > 1:
                    color = (
                        (255, 0, 0)
                        if area_name == "display_check_area_points"
                        else (0, 0, 255)
                    )  # 青または赤
                    cv2.polylines(
                        display_frame,
                        [np.array(pixel_points)],
                        isClosed=True,
                        color=color,
                        thickness=2,
                    )
                    # エリア名表示
                    cv2.putText(
                        display_frame,
                        area_name,
                        (pixel_points[0][0] + 10, pixel_points[0][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

        # 現在設定中のエリア名を表示
        cv2.putText(
            display_frame,
            f"Setting: {current_area_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("ROI Setup", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting without saving current changes.")
            break
        elif key == ord("s"):
            areas_to_define[
                current_area_name
            ] = drawing_points  # 現在の作業内容をareas_to_defineに反映
            save_roi_config(frame_width, frame_height)
            break  # 保存して終了
        elif key == ord("1"):
            areas_to_define[current_area_name] = drawing_points  # 現在の作業内容を保存
            current_area_name = "roi_points"
            drawing_points = areas_to_define[current_area_name][:]  # 選択エリアの既存点を読み込み
            print(f"Switched to setting: {current_area_name}")
        elif key == ord("2"):
            areas_to_define[current_area_name] = drawing_points
            current_area_name = "display_check_area_points"
            drawing_points = areas_to_define[current_area_name][:]
            print(f"Switched to setting: {current_area_name}")
        elif key == ord("3"):
            areas_to_define[current_area_name] = drawing_points
            current_area_name = "clear_roi_points"
            drawing_points = areas_to_define[current_area_name][:]
            print(f"Switched to setting: {current_area_name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
