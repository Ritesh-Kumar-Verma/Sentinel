import os

import cv2
import numpy as np

from face_encoder import FaceEncoder


DATASET_PATH = "../../dataset"
DATABASE_PATH = "../../database"


def main():
    os.makedirs(DATABASE_PATH, exist_ok=True)

    encoder = FaceEncoder()

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        # Skip files. We only want person directories.
        if not os.path.isdir(person_path):
            continue

        print(f"\nProcessing: {person_name}")

        image_files = os.listdir(person_path)

        embeddings = []

        for image_name in image_files:
            image_path = os.path.join(
                person_path,
                image_name,
            )

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            try:
                embedding = encoder.encode(image)
                embeddings.append(embedding)

                print(
                    f"  Encoded: {image_name}"
                )

            except ValueError as error:
                print(
                    f"  Skipped {image_name}: {error}"
                )

        if not embeddings:
            print(
                f"No valid faces found for {person_name}"
            )
            continue

        # If there are multiple images, calculate
        # the average embedding for this person.
        person_embedding = np.mean(
            embeddings,
            axis=0,
        )

        output_path = os.path.join(
            DATABASE_PATH,
            f"{person_name}.npy",
        )

        np.save(
            output_path,
            person_embedding,
        )

        print(
            f"Saved embedding: {output_path}"
        )


if __name__ == "__main__":
    main()