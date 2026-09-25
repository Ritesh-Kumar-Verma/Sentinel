import cv2
import streamlit as st
import os
from face_detector import FaceDetector
from face_encoder import FaceEncoder
import numpy as np


IMAGE_PATH = "../../test_images/t2.png"
OUTPUT_PATH = "../../results/detection_result.jpg"
DATABASE_PATH = "G:/AI/Sentinel/database"

def load_database():
    """
    Load all face embeddings from the database.

    Returns:
        dict:
            {
                "Name1": embedding,
                "Name2": embedding
            }
    """

    database = {}

    for filename in os.listdir(DATABASE_PATH):
        if not filename.endswith(".npy"):
            continue

        person_name = os.path.splitext(filename)[0]

        path = os.path.join(
            DATABASE_PATH,
            filename,
        )

        embedding = np.load(path)

        database[person_name] = embedding

    return database

def recognize_face(
    embedding,
    database,
    threshold=0.5,
):
    """
    Compare a face embedding against the database.

    Returns:
        person_name, similarity
    """

    best_match = None
    best_similarity = -1.0

    for person_name, stored_embedding in database.items():

        similarity = cosine_similarity(
            embedding,
            stored_embedding,
        )

        print(
            f"{person_name}: "
            f"{similarity:.4f}"
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_name

    if best_similarity < threshold:
        return "Unknown", best_similarity

    return best_match, best_similarity
def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two embeddings.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b))
    )


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


def recognize_frame(frame, threshold=0.5):
    """
    Detect and recognize faces in a frame.

    Args:
        frame: OpenCV BGR frame.
        encoder: FaceEncoder instance.
        database: Face embedding database.
        threshold: Minimum cosine similarity for recognition.

    Returns:
        frame: Frame with bounding boxes and identity labels drawn.
    """
    database = load_database()
    encoder = FaceEncoder()

    if frame is None:
        return frame
    
    
    # print("database=========================",database)
    
    # Detect faces
    faces = encoder.app.get(frame)

    for face in faces:

        # Get face embedding
        embedding = face.embedding

        # Recognize face
        name, similarity = recognize_face(
            embedding,
            database,
            threshold=threshold,
        )

        # Get bounding box
        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Choose color
        if name == "Unknown":
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # Label
        label = name

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return frame

if __name__ == "__main__":
    # face_detect()
    frame = recognize_frame(cv2.imread(IMAGE_PATH))
    cv2.imwrite(OUTPUT_PATH,frame)
    