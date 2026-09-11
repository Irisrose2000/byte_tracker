import numpy as np

from bytetrack import BYTETracker


class MultiClassByteTracker:

    def __init__(
        self,
        class_names,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        low_thresh=0.1,
    ):

        self.class_names = class_names

        self.trackers = {}

        for class_name in class_names:

            self.trackers[class_name] = BYTETracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                low_thresh=low_thresh,
            )

    def update(self, detections):

        results = []

        # Process each class separately
        for class_name in self.class_names:

            class_detections = [
                d
                for d in detections
                if d["class_name"] == class_name
            ]

            if len(class_detections) == 0:

                detection_array = np.empty(
                    (0, 5),
                    dtype=np.float32,
                )

            else:

                detection_array = np.array(
                    [
                        [
                            d["bbox"][0],
                            d["bbox"][1],
                            d["bbox"][2],
                            d["bbox"][3],
                            d["confidence"],
                        ]
                        for d in class_detections
                    ],
                    dtype=np.float32,
                )

            tracks = self.trackers[
                class_name
            ].update(
                detection_array
            )

            for track in tracks:

                x1 = float(track[0])
                y1 = float(track[1])
                x2 = float(track[2])
                y2 = float(track[3])

                track_id = int(track[4])

                results.append(
                    {
                        "class_name": class_name,
                        "track_id": track_id,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }
                )

        return results
