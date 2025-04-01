import time

import cv2 as cv
import numpy as np
import pandas as pd
import xarray as xr


class RecordWorker:
    def __init__(self, vid_dir, data_dir, landmark_labels, time_axis="frame"):
        super().__init__()

        self.vid_dir = vid_dir
        self.data_dir = data_dir
        self.landmark_labels = landmark_labels
        self.time_axis = time_axis

        self.save_name = None

        self.frame_no = 0
        self.fps = 30

        self.video_writer = None
        self.start_time = time.time()
        self.pose = []
        self.pose_world = []
        self.params = []
        self.timestamps = []
        self.frames = []
        self.recording = False

    def start_record(self, save_name, frame_shape=None):
        self.save_name = save_name
        self.start_time = time.time()
        self.recording = True

        if frame_shape is not None:
            vid_path = self.vid_dir / f"{save_name}.mp4"
            fourcc = cv.VideoWriter_fourcc(*"MP4V")
            self.video_writer = cv.VideoWriter(
                str(vid_path),
                fourcc,
                self.fps,
                (frame_shape[1], frame_shape[0]),
                isColor=True,
            )

    def record_data(
        self, timestamp_ms, pose, pose_world=None, frame=None, params=None, scores=None
    ):
        if self.recording:
            self.frame_no += 1
            if pose is not None:
                self.frames.append(self.frame_no)
                self.timestamps.append(timestamp_ms)
                self.pose.append(pose)
            if pose_world is not None:
                self.pose_world.append(pose_world)
            if params is not None:
                if scores is not None:
                    params.update(scores)  # スコアをparamsに追加
                self.params.append(params)
            if frame is not None:
                self.video_writer.write(frame)

    def end_record(self):
        if not self.recording:
            return

        self.recording = False
        self.video_writer.release()
        self.video_writer = None

        if not self.pose:
            return

        axis = ["x", "y", "z", "visibility", "presence"]
        markers_description = list(self.landmark_labels.keys())
        if self.time_axis == "frame":
            da = xr.DataArray(
                np.array(self.pose),
                dims=["frame", "marker", "axis"],
                coords={
                    "frame": self.frames,
                    "marker": markers_description,
                    "axis": axis,
                },
                name="landmarks",
                attrs={"fps": self.fps},
            )
            df = pd.DataFrame(self.params, index=self.frames)
        elif self.time_axis == "time":
            da = xr.DataArray(
                np.array(self.pose),
                dims=["time", "marker", "axis"],
                coords={
                    "time": self.timestamps,
                    "marker": markers_description,
                    "axis": axis,
                },
                name="landmarks",
                attrs={"fps": self.fps},
            )
            df = pd.DataFrame(self.params, index=self.timestamps)
        else:
            raise ValueError("time_axis must be 'frame' or 'time'")

        da.to_dataframe().to_csv(self.data_dir / f"{self.save_name}.csv")
        da.to_netcdf(self.data_dir / f"{self.save_name}.nc")
        df.to_csv(self.data_dir / f"{self.save_name}_params.csv")  # スコアも含めて保存

        self.frame_no = 0
        self.pose = []
        self.params = []
        self.frames = []
        self.timestamps = []
