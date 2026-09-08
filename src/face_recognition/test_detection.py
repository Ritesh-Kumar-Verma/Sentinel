import cv2

from face_detector import FaceDetector


IMAGE_PATH = "../../test_images/Tony.png"
OUTPUT_PATH = "../../results/detection_result.jpg"




def main():
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
    main()
 