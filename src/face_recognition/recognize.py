import os

import cv2
import numpy as np

from face_encoder import FaceEncoder


DATABASE_PATH = "../../database"
IMAGE_PATH = "../../test_images/2.jpeg"


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


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    print("Loading face database...")

    database = load_database()

    if not database:
        raise RuntimeError(
            "Face database is empty."
        )

    print(
        f"Loaded {len(database)} identities."
    )

    encoder = FaceEncoder()

    print("Detecting faces...")

    faces = encoder.app.get(image)

    print(
        f"Number of faces detected: {len(faces)}"
    )

    for i, face in enumerate(faces):

        embedding = face.embedding

        name, similarity = recognize_face(
            embedding,
            database,
        )

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        print(
            f"\nFace {i + 1}:"
        )

        print(
            f"Identity: {name}"
        )

        print(
            f"Similarity: {similarity:.4f}"
        )

        # Draw bounding box.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Draw identity.
        label = (
            f"{name} "
            f"({similarity:.2f})"
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    output_path = "../../results/recognition_result.jpg"

    cv2.imwrite(
        output_path,
        image,
    )

    print(
        f"\nResult saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
