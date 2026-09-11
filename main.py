import cv2
import os

from qwen_detector import QwenDetector
from bytetrack_wrapper import MultiClassByteTracker


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_VIDEO = "input/video.mp4"

OUTPUT_VIDEO = "output/tracked.mp4"

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

TARGET_CLASSES = [
    "person",
    # "basketball",
    # "referee",
]

# Qwen detection threshold
DETECTION_CONFIDENCE = 0.20

# BYTETrack parameters
TRACK_THRESH = 0.5

TRACK_BUFFER = 30

MATCH_THRESH = 0.8

LOW_THRESH = 0.1

# Process every frame
FRAME_SKIP = 1


# ============================================================
# COLORS
# ============================================================

COLORS = {
    "person": (0, 255, 0),
    "basketball": (0, 165, 255),
    "referee": (255, 0, 0),
}


# ============================================================
# DRAW FUNCTION
# ============================================================

def draw_tracks(
    frame,
    tracks,
):

    for track in tracks:

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        track_id = track["track_id"]

        class_name = track["class_name"]

        color = COLORS.get(
            class_name,
            (255, 255, 255)
        )

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # Label
        label = (
            f"{class_name} "
            f"ID:{track_id}"
        )

        # Text size
        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        # Background rectangle
        cv2.rectangle(
            frame,
            (
                x1,
                max(
                    0,
                    y1 - text_height - baseline
                ),
            ),
            (
                x1 + text_width,
                y1,
            ),
            color,
            -1,
        )

        # Text
        cv2.putText(
            frame,
            label,
            (
                x1,
                y1 - baseline,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    # Create output directory
    os.makedirs(
        "output",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load Qwen
    # --------------------------------------------------------

    detector = QwenDetector(
        model_name=MODEL_NAME
    )

    # --------------------------------------------------------
    # Create BYTETrack
    # --------------------------------------------------------

    tracker = MultiClassByteTracker(
        class_names=TARGET_CLASSES,
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        low_thresh=LOW_THRESH,
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {INPUT_VIDEO}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video: {width}x{height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Frames: {total_frames}"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height),
    )

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        print(
            f"\rProcessing frame "
            f"{frame_id}/{total_frames}",
            end="",
        )

        # ----------------------------------------------------
        # Qwen detection
        # ----------------------------------------------------

        detections = detector.detect(
            frame,
            target_classes=TARGET_CLASSES,
            confidence_threshold=DETECTION_CONFIDENCE,
        )

        # ----------------------------------------------------
        # BYTETrack
        # ----------------------------------------------------

        tracks = tracker.update(
            detections
        )

        # ----------------------------------------------------
        # Draw results
        # ----------------------------------------------------

        output_frame = draw_tracks(
            frame,
            tracks,
        )

        # ----------------------------------------------------
        # Display FPS / frame number
        # ----------------------------------------------------

        cv2.putText(
            output_frame,
            f"Frame: {frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        writer.write(
            output_frame
        )

        # ----------------------------------------------------
        # Optional live display
        # ----------------------------------------------------

        cv2.imshow(
            "Qwen + BYTETrack",
            output_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print()

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print(
        f"Finished."
    )

    print(
        f"Output saved to: {OUTPUT_VIDEO}"
    )


if __name__ == "__main__":

    main()
