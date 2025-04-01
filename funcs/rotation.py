import numpy as np
import cv2


# rotate a markers corners by rvec and translate by tvec if given
# input is the size of a marker.
# In the markerworld the 4 markercorners are at (x,y) = (+- markersize/2, +- markersize/2)
# returns the rotated and translated corners and the rotation matrix
def rotate_marker_corners(rvec, markersize, tvec=None):
    mhalf = markersize / 2.0
    # convert rot vector to rot matrix both do: markerworld -> cam-world
    mrv, jacobian = cv2.Rodrigues(rvec)

    # in markerworld the corners are all in the xy-plane so z is zero at first
    X = mhalf * mrv[:, 0]  # rotate the x = mhalf
    Y = mhalf * mrv[:, 1]  # rotate the y = mhalf
    minusX = X * (-1)
    minusY = Y * (-1)

    # calculate 4 corners of the marker in camworld. corners are enumerated clockwise
    markercorners = []
    markercorners.append(np.add(minusX, Y))  # was upper left in markerworld
    markercorners.append(np.add(X, Y))  # was upper right in markerworld
    markercorners.append(np.add(X, minusY))  # was lower right in markerworld
    markercorners.append(np.add(minusX, minusY))  # was lower left in markerworld
    # if tvec given, move all by tvec
    if tvec is not None:
        C = tvec  # center of marker in camworld
        for i, mc in enumerate(markercorners):
            markercorners[i] = np.add(C, mc)  # add tvec to each corner
    # print('Vec X, Y, C, dot(X,Y)', X,Y,C, np.dot(X,Y)) # just for debug
    markercorners = np.array(
        markercorners, dtype=np.float32
    )  # type needed when used as input to cv2
    return markercorners, mrv


"""
Copyright 2019 Marco Noll, Garmin International Inc. Licensed under the Apache 
License, Version 2.0 (the "License"); you may not use this file except in compliance 
with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific 
language governing permissions and limitations under the License.
"""
