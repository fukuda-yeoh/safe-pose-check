import cv2 as cv
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def get_landmark_labels(landmarks_file):
    labels = {}
    with open(landmarks_file, "r") as f:
        rows = f.readlines()
    for row in rows:
        k, v = row.strip().split(" - ")
        labels[v] = int(k)
    return labels


def get_pos(landmarks, labels, landmark_name, body=0):
    """Get position of landmark using label name"""
    i = labels[landmark_name]
    return landmarks[body][i]


def get_pos_array(landmark_array, labels, landmark_name):
    """Get position of landmark using label name"""
    i = labels[landmark_name]
    return landmark_array[i, :]


def landmarks_to_array(landmarks, body=0):
    landmark_list = landmarks[body]
    np_arr = np.zeros(shape=(len(landmark_list), 5), dtype=float)
    for i, _landmark in enumerate(landmark_list):
        np_arr[i, 0] = _landmark.x
        np_arr[i, 1] = _landmark.y
        np_arr[i, 2] = _landmark.z
        np_arr[i, 3] = _landmark.visibility
        np_arr[i, 4] = _landmark.presence
    return np_arr


def draw_landmarks_on_image(rgb_image, pose_landmarks_list):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def draw_arrow_on_image(image, start, end, color=(0, 255, 0), thickness=3):
    annotated_image = np.copy(image)

    height, width, _ = image.shape
    start_point = np.array([start[0] * width, start[1] * height], dtype=int)
    end_point = np.array([end[0] * width, end[1] * height], dtype=int)
    thickness = int(thickness)
    cv.arrowedLine(
        annotated_image, tuple(start_point), tuple(end_point), color, thickness
    )
    return annotated_image


def draw_roi_on_image(image, roi_points, color=(0, 255, 0), thickness=2):
    annotated_image = np.copy(image)
    height, width, _ = image.shape
    roi_points_int = [(int(x * width), int(y * height)) for (x, y) in roi_points]
    # ポリゴンとして描画する
    cv.polylines(
        annotated_image,
        [np.array(roi_points_int)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )
    return annotated_image


def draw_clear_roi_on_image(image, clear_roi_points, color=(0, 0, 255), thickness=2):
    annotated_image = np.copy(image)
    height, width, _ = image.shape
    clear_roi_points_int = [
        (int(x * width), int(y * height)) for (x, y) in clear_roi_points
    ]
    # ポリゴンとして描画する
    cv.polylines(
        annotated_image,
        [np.array(clear_roi_points_int)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )
    return annotated_image
