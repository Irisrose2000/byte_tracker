import json
import re
import torch

from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


class QwenDetector:

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        device="auto",
    ):

        print("Loading Qwen model...")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        print("Qwen loaded.")

    def detect(
        self,
        frame,
        target_classes=None,
        confidence_threshold=0.20,
    ):
        """
        frame:
            OpenCV BGR image

        Returns:
            [
                {
                    "class_name": "person",
                    "class_id": 0,
                    "confidence": 0.91,
                    "bbox": [x1, y1, x2, y2]
                }
            ]
        """

        if target_classes is None:
            target_classes = ["person"]

        # Convert BGR -> RGB
        frame_rgb = frame[:, :, ::-1]

        image = Image.fromarray(frame_rgb)

        height, width = frame.shape[:2]

        classes_string = ", ".join(target_classes)

        prompt = f"""
You are an object detection system.

Detect ONLY these object classes:

{classes_string}

Return every detected object.

For every object return:

- class
- confidence between 0 and 1
- bounding box [x1, y1, x2, y2]

Coordinates must be pixel coordinates relative to the image.

The coordinate origin is the top-left corner.

x1 < x2
y1 < y2

Return ONLY valid JSON.

Required format:

[
  {{
    "class": "person",
    "confidence": 0.95,
    "bbox": [100, 50, 250, 400]
  }}
]

If there are no objects, return:

[]
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # Prepare Qwen input
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        # Run Qwen
        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
            )

        # Remove prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(
                inputs.input_ids,
                generated_ids
            )
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        detections = self.parse_output(
            output_text,
            width,
            height,
            target_classes,
            confidence_threshold,
        )

        return detections

    def parse_output(
        self,
        output_text,
        width,
        height,
        target_classes,
        confidence_threshold,
    ):

        # Remove markdown code blocks if Qwen returns them
        output_text = output_text.strip()

        output_text = re.sub(
            r"```json",
            "",
            output_text,
            flags=re.IGNORECASE,
        )

        output_text = re.sub(
            r"```",
            "",
            output_text,
        )

        output_text = output_text.strip()

        # Find JSON array
        match = re.search(
            r"\[.*\]",
            output_text,
            flags=re.DOTALL,
        )

        if not match:
            print(
                "Could not find JSON in Qwen response:"
            )
            print(output_text)

            return []

        json_text = match.group(0)

        try:
            objects = json.loads(json_text)

        except json.JSONDecodeError:

            print(
                "Invalid JSON from Qwen:"
            )

            print(output_text)

            return []

        detections = []

        class_to_id = {
            name.lower(): i
            for i, name in enumerate(target_classes)
        }

        for obj in objects:

            try:

                class_name = str(
                    obj["class"]
                ).lower()

                confidence = float(
                    obj["confidence"]
                )

                bbox = obj["bbox"]

                if class_name not in class_to_id:
                    continue

                if confidence < confidence_threshold:
                    continue

                if len(bbox) != 4:
                    continue

                x1 = float(bbox[0])
                y1 = float(bbox[1])
                x2 = float(bbox[2])
                y2 = float(bbox[3])

                # Clamp coordinates
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1:
                    continue

                if y2 <= y1:
                    continue

                detections.append(
                    {
                        "class_name": class_name,
                        "class_id": class_to_id[
                            class_name
                        ],
                        "confidence": confidence,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }
                )

            except Exception as e:

                print(
                    "Error parsing detection:",
                    e
                )

        return detections
