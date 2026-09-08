import cv2
import streamlit as st

from face_detector import FaceDetector
from recognize import recognize_frame


IMAGE_PATH = "../../test_images/image.png"
OUTPUT_PATH = "../../results/detection_result.jpg"


def face_detect():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    detector = FaceDetector()

    faces = detector.detect(image)

    print(f"Number of faces detected: {len(faces)}")

    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        print(
            f"Face {i + 1}: "
            f"({x1}, {y1}) -> ({x2}, {y2})"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    cv2.imwrite(OUTPUT_PATH, image)

    print(f"Result saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    # face_detect()
    frame = recognize_frame(cv2.imread(IMAGE_PATH))
    cv2.imwrite(OUTPUT_PATH,frame)
    