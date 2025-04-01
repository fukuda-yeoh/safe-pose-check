import json

import cv2 as cv
import numpy as np


def load_intrinsic(file_path):
    with open(file_path, mode="r") as f:
        cal_data = json.load(f)

    if "fisheye" in cal_data:
        fisheye = cal_data["fisheye"]
    else:
        fisheye = False

    intrinsic_matrix = np.array(cal_data["K"])
    distortion_coeffs = np.asarray(cal_data["D"], dtype=float)

    return intrinsic_matrix, distortion_coeffs, fisheye


def get_undistort_funcs(
    shape, intrinsic_matrix, distortion_coeffs, fisheye=False, scale=1.0
):
    h, w = shape[:2]

    if fisheye:
        newcameramtx = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intrinsic_matrix, distortion_coeffs, (w, h), np.eye(3), balance=0
        )
        newcameramtx[0, 0] = newcameramtx[0, 0] * scale
        newcameramtx[1, 1] = newcameramtx[1, 1] * scale

        map_x, map_y = cv.fisheye.initUndistortRectifyMap(
            intrinsic_matrix,
            distortion_coeffs,
            np.eye(3),
            newcameramtx,
            (w, h),
            cv.CV_16SC2,
        )
    else:
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            intrinsic_matrix,
            distortion_coeffs,
            (w, h),
            1,
            (w, h),
        )
        newcameramtx[0, 0] = newcameramtx[0, 0] * scale
        newcameramtx[1, 1] = newcameramtx[1, 1] * scale

        map_x, map_y = cv.initUndistortRectifyMap(
            intrinsic_matrix,
            distortion_coeffs,
            np.eye(3),
            newcameramtx,
            (w, h),
            cv.CV_16SC2,
        )

    return map_x, map_y


def undistort_map(img, map_x, map_y):
    return cv.remap(img, map_x, map_y, cv.INTER_LINEAR)


def undistort_points(
    points, intrinsic_matrix, distortion_coeffs, newcameramtx, fisheye=False
):
    if fisheye:
        return cv.fisheye.undistortPoints(
            points, intrinsic_matrix, distortion_coeffs, None, newcameramtx
        )
    else:
        return cv.undistortPoints(
            points, intrinsic_matrix, distortion_coeffs, None, newcameramtx
        )


def redistort_points(
    points, intrinsic_matrix, distortion_coeffs, newcameramtx, fisheye=False
):
    scaled_points = np.vstack(
        [
            (points[:, 0] - newcameramtx[0, 2]) / newcameramtx[0, 0],
            (points[:, 1] - newcameramtx[1, 2]) / newcameramtx[1, 1],
            np.zeros(points.shape[0]),
        ]
    ).T
    if fisheye:
        distorted_points, _ = cv.fisheye.projectPoints(
            scaled_points,
            np.zeros(3),
            np.zeros(3),
            intrinsic_matrix,
            distortion_coeffs,
        )
    else:
        distorted_points, _ = cv.projectPoints(
            scaled_points,
            np.zeros(3),
            np.zeros(3),
            intrinsic_matrix,
            distortion_coeffs,
            aspectRatio=newcameramtx[0, 0] / newcameramtx[1, 1],
        )
    return distorted_points
