import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# Utility functions
# ============================================================

def iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Box format:
        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(
        0.0,
        box1[2] - box1[0]
    ) * max(
        0.0,
        box1[3] - box1[1]
    )

    area2 = max(
        0.0,
        box2[2] - box2[0]
    ) * max(
        0.0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def iou_matrix(tracks, detections):
    """
    Create IoU matrix.

    tracks:
        list of Track objects

    detections:
        numpy array [N, 5]
        [x1, y1, x2, y2, confidence]
    """

    if len(tracks) == 0 or len(detections) == 0:
        return np.empty(
            (len(tracks), len(detections))
        )

    matrix = np.zeros(
        (len(tracks), len(detections)),
        dtype=np.float32
    )

    for i, track in enumerate(tracks):

        for j, detection in enumerate(detections):

            matrix[i, j] = iou(
                track.bbox,
                detection[:4]
            )

    return matrix


# ============================================================
# Simple Kalman Filter
# ============================================================

class KalmanFilter:

    def __init__(self):

        # State:
        #
        # x
        # y
        # aspect ratio
        # height
        # vx
        # vy
        # va
        # vh
        #

        self.ndim = 4
        self.dt = 1.0

        self._motion_mat = np.eye(
            2 * self.ndim,
            dtype=np.float32
        )

        for i in range(self.ndim):

            self._motion_mat[
                i,
                self.ndim + i
            ] = self.dt

        self._update_mat = np.eye(
            self.ndim,
            2 * self.ndim,
            dtype=np.float32
        )

    def initiate(self, measurement):

        """
        Initialize track.

        measurement:
            [center_x, center_y, aspect_ratio, height]
        """

        mean_pos = measurement

        mean_vel = np.zeros_like(
            mean_pos
        )

        mean = np.r_[
            mean_pos,
            mean_vel
        ]

        std = np.array([
            2.0,
            2.0,
            1e-2,
            2.0,
            10.0,
            10.0,
            1e-3,
            10.0
        ], dtype=np.float32)

        covariance = np.diag(
            std ** 2
        )

        return mean.astype(
            np.float32
        ), covariance.astype(
            np.float32
        )

    def predict(
        self,
        mean,
        covariance
    ):

        mean = np.dot(
            self._motion_mat,
            mean
        )

        covariance = (
            self._motion_mat
            @ covariance
            @ self._motion_mat.T
        )

        return mean, covariance

    def update(
        self,
        mean,
        covariance,
        measurement
    ):

        projected_mean = (
            self._update_mat
            @ mean
        )

        projected_covariance = (
            self._update_mat
            @ covariance
            @ self._update_mat.T
        )

        innovation_covariance = (
            projected_covariance
            + np.eye(4) * 1e-2
        )

        kalman_gain = (
            covariance
            @ self._update_mat.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )

        innovation = (
            measurement
            - projected_mean
        )

        new_mean = (
            mean
            + kalman_gain @ innovation
        )

        new_covariance = (
            covariance
            - kalman_gain
            @ self._update_mat
            @ covariance
        )

        return (
            new_mean.astype(np.float32),
            new_covariance.astype(np.float32)
        )


# ============================================================
# Track
# ============================================================

class Track:

    _next_id = 1

    def __init__(
        self,
        bbox,
        confidence,
        kalman_filter
    ):

        self.bbox = np.asarray(
            bbox,
            dtype=np.float32
        )

        self.confidence = float(
            confidence
        )

        self.track_id = Track._next_id

        Track._next_id += 1

        self.kalman_filter = (
            kalman_filter
        )

        self.mean = None
        self.covariance = None

        self.age = 0
        self.time_since_update = 0

        self.hits = 1

        self.confirmed = True

        self._initialize()

    def _initialize(self):

        measurement = (
            self.bbox_to_measurement(
                self.bbox
            )
        )

        (
            self.mean,
            self.covariance
        ) = self.kalman_filter.initiate(
            measurement
        )

    @staticmethod
    def bbox_to_measurement(
        bbox
    ):

        x1, y1, x2, y2 = bbox

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        aspect_ratio = (
            width / height
        )

        return np.array(
            [
                center_x,
                center_y,
                aspect_ratio,
                height
            ],
            dtype=np.float32
        )

    @staticmethod
    def measurement_to_bbox(
        measurement
    ):

        center_x = measurement[0]
        center_y = measurement[1]

        aspect_ratio = max(
            1e-3,
            measurement[2]
        )

        height = max(
            1.0,
            measurement[3]
        )

        width = (
            aspect_ratio
            * height
        )

        x1 = (
            center_x
            - width / 2
        )

        y1 = (
            center_y
            - height / 2
        )

        x2 = (
            center_x
            + width / 2
        )

        y2 = (
            center_y
            + height / 2
        )

        return np.array(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dtype=np.float32
        )

    def predict(self):

        (
            self.mean,
            self.covariance
        ) = self.kalman_filter.predict(
            self.mean,
            self.covariance
        )

        self.bbox = (
            self.measurement_to_bbox(
                self.mean
            )
        )

        self.age += 1

        self.time_since_update += 1

    def update(
        self,
        bbox,
        confidence
    ):

        measurement = (
            self.bbox_to_measurement(
                bbox
            )
        )

        (
            self.mean,
            self.covariance
        ) = self.kalman_filter.update(
            self.mean,
            self.covariance,
            measurement
        )

        self.bbox = (
            self.measurement_to_bbox(
                self.mean
            )
        )

        self.confidence = float(
            confidence
        )

        self.time_since_update = 0

        self.hits += 1


# ============================================================
# BYTETracker
# ============================================================

class BYTETracker:

    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        low_thresh=0.1
    ):

        """
        Parameters
        ----------

        track_thresh:
            Confidence threshold for high-confidence
            detections.

        track_buffer:
            Number of frames a lost track is kept.

        match_thresh:
            IoU threshold for association.

        low_thresh:
            Lower confidence threshold.
        """

        self.track_thresh = float(
            track_thresh
        )

        self.track_buffer = int(
            track_buffer
        )

        self.match_thresh = float(
            match_thresh
        )

        self.low_thresh = float(
            low_thresh
        )

        self.kalman_filter = (
            KalmanFilter()
        )

        self.tracked_tracks = []

        self.lost_tracks = []

        self.frame_id = 0

    # --------------------------------------------------------
    # Predict existing tracks
    # --------------------------------------------------------

    def predict(self):

        for track in self.tracked_tracks:

            track.predict()

        for track in self.lost_tracks:

            track.predict()

    # --------------------------------------------------------
    # Associate tracks with detections
    # --------------------------------------------------------

    def associate(
        self,
        tracks,
        detections,
        match_threshold
    ):

        if (
            len(tracks) == 0
            or len(detections) == 0
        ):

            return [], list(
                range(len(tracks))
            ), list(
                range(len(detections))
            )

        cost_matrix = (
            1.0
            - iou_matrix(
                tracks,
                detections
            )
        )

        row_indices, col_indices = (
            linear_sum_assignment(
                cost_matrix
            )
        )

        matches = []

        unmatched_tracks = set(
            range(len(tracks))
        )

        unmatched_detections = set(
            range(len(detections))
        )

        for row, col in zip(
            row_indices,
            col_indices
        ):

            current_iou = (
                1.0
                - cost_matrix[row, col]
            )

            if current_iou >= match_threshold:

                matches.append(
                    (row, col)
                )

                unmatched_tracks.discard(
                    row
                )

                unmatched_detections.discard(
                    col
                )

        return (
            matches,
            list(unmatched_tracks),
            list(unmatched_detections)
        )

    # --------------------------------------------------------
    # Main update
    # --------------------------------------------------------

    def update(
        self,
        detections
    ):

        """
        Update BYTETrack.

        Input:

            numpy array with shape [N, 5]

            [
                x1,
                y1,
                x2,
                y2,
                confidence
            ]

        Returns:

            numpy array with shape [N, 5]

            [
                x1,
                y1,
                x2,
                y2,
                track_id
            ]
        """

        self.frame_id += 1

        # ----------------------------------------------------
        # Convert detections
        # ----------------------------------------------------

        if detections is None:

            detections = np.empty(
                (0, 5),
                dtype=np.float32
            )

        detections = np.asarray(
            detections,
            dtype=np.float32
        )

        if detections.size == 0:

            detections = np.empty(
                (0, 5),
                dtype=np.float32
            )

        if detections.ndim == 1:

            detections = detections.reshape(
                -1,
                5
            )

        # ----------------------------------------------------
        # Split high and low confidence detections
        # ----------------------------------------------------

        high_detections = detections[
            detections[:, 4]
            >= self.track_thresh
        ]

        low_detections = detections[
            (
                detections[:, 4]
                >= self.low_thresh
            )
            &
            (
                detections[:, 4]
                < self.track_thresh
            )
        ]

        # ----------------------------------------------------
        # Predict tracks
        # ----------------------------------------------------

        self.predict()

        # ----------------------------------------------------
        # First association
        #
        # High confidence detections
        # ----------------------------------------------------

        (
            matches,
            unmatched_tracks,
            unmatched_high
        ) = self.associate(
            self.tracked_tracks,
            high_detections,
            self.match_thresh
        )

        # Update matched tracks
        for track_idx, detection_idx in matches:

            detection = (
                high_detections[
                    detection_idx
                ]
            )

            self.tracked_tracks[
                track_idx
            ].update(
                detection[:4],
                detection[4]
            )

        # ----------------------------------------------------
        # Second association
        #
        # Low confidence detections
        # ----------------------------------------------------

        remaining_tracks = [
            self.tracked_tracks[i]
            for i in unmatched_tracks
        ]

        (
            low_matches,
            still_unmatched_tracks,
            unmatched_low
        ) = self.associate(
            remaining_tracks,
            low_detections,
            0.3
        )

        # Update tracks matched with low-score detections
        for local_track_idx, detection_idx in low_matches:

            track = remaining_tracks[
                local_track_idx
            ]

            detection = (
                low_detections[
                    detection_idx
                ]
            )

            track.update(
                detection[:4],
                detection[4]
            )

        # ----------------------------------------------------
        # Determine tracks that are lost
        # ----------------------------------------------------

        lost_track_objects = []

        for local_idx in still_unmatched_tracks:

            track = remaining_tracks[
                local_idx
            ]

            lost_track_objects.append(
                track
            )

        # Remove lost tracks from active list
        active_ids = {
            id(track)
            for track in self.tracked_tracks
            if track not in lost_track_objects
        }

        self.tracked_tracks = [
            track
            for track in self.tracked_tracks
            if id(track) in active_ids
        ]

        # Move lost tracks to lost list
        for track in lost_track_objects:

            if track not in self.lost_tracks:

                self.lost_tracks.append(
                    track
                )

        # ----------------------------------------------------
        # Try to recover lost tracks
        #
        # This allows short-term occlusion recovery.
        # ----------------------------------------------------

        if len(self.lost_tracks) > 0:

            (
                lost_matches,
                unmatched_lost,
                unmatched_high_recovery
            ) = self.associate(
                self.lost_tracks,
                high_detections[
                    unmatched_high
                ],
                0.3
            )

            for (
                lost_idx,
                detection_idx
            ) in lost_matches:

                track = self.lost_tracks[
                    lost_idx
                ]

                detection = (
                    high_detections[
                        unmatched_high[
                            detection_idx
                        ]
                    ]
                )

                track.update(
                    detection[:4],
                    detection[4]
                )

                self.tracked_tracks.append(
                    track
                )

            # Keep only genuinely lost tracks
            recovered_indices = {
                x[0]
                for x in lost_matches
            }

            self.lost_tracks = [
                track
                for idx, track in enumerate(
                    self.lost_tracks
                )
                if idx not in recovered_indices
            ]

        # ----------------------------------------------------
        # Create new tracks
        #
        # Only high confidence detections create
        # new identities.
        # ----------------------------------------------------

        for detection_idx in unmatched_high:

            detection = (
                high_detections[
                    detection_idx
                ]
            )

            # BYTETrack's important rule:
            # low-confidence detections do NOT
            # create new tracks.

            if (
                detection[4]
                >= self.track_thresh
            ):

                new_track = Track(
                    detection[:4],
                    detection[4],
                    self.kalman_filter
                )

                self.tracked_tracks.append(
                    new_track
                )

        # ----------------------------------------------------
        # Remove old lost tracks
        # ----------------------------------------------------

        self.lost_tracks = [
            track
            for track in self.lost_tracks
            if track.time_since_update
            <= self.track_buffer
        ]

        # ----------------------------------------------------
        # Prepare output
        # ----------------------------------------------------

        results = []

        for track in self.tracked_tracks:

            if (
                track.time_since_update
                == 0
            ):

                results.append(
                    [
                        track.bbox[0],
                        track.bbox[1],
                        track.bbox[2],
                        track.bbox[3],
                        track.track_id
                    ]
                )

        if len(results) == 0:

            return np.empty(
                (0, 5),
                dtype=np.float32
            )

        return np.asarray(
            results,
            dtype=np.float32
        )


# ============================================================
# Reset IDs
# ============================================================

def reset_track_ids():

    Track._next_id = 1
