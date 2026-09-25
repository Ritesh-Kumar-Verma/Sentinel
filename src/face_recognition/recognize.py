import os

import cv2
import numpy as np

from .face_encoder import FaceEncoder


DATABASE_PATH = "G:/AI/Sentinel/database"
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


def recognize_frame(frame,database,encoder, threshold=0.5):
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






def recognize_frame_seprate(frame, database, unknown_database, encoder, threshold=0.5):
    """
    Returns:
        dict: {"Name": cropped_face_image} for all faces found in the frame.
    """
    if frame is None:
        return {}
    
    height, width = frame.shape[:2]
    margin = 30 
    
    # Dictionary to hold the faces we find in THIS specific frame
    faces_in_frame = {}
    
    # Detect faces
    faces = encoder.app.get(frame)

    for face in faces:
        embedding = face.embedding
        
        # 1. First, check if they are in the KNOWN database
        name, similarity = recognize_face(embedding, database, threshold=threshold)

        # 2. If they are Unknown, check if we've seen this specific unknown before
        if name == "Unknown":
            # Re-use your exact same recognize_face function, but on the strangers!
            unk_name, unk_sim = recognize_face(embedding, unknown_database, threshold=threshold)
            
            if unk_name == "Unknown":
                # Brand new stranger! Assign them a number and save their embedding
                new_id = len(unknown_database) + 1
                name = f"Unknown_{new_id}"
                unknown_database[name] = embedding
            else:
                # We've seen this stranger before, use their assigned unknown number
                name = unk_name

        # Choose color
        if "Unknown" in name:
            color = (0, 0, 255)  # Red for all unknowns
        else:
            color = (0, 255, 0)  # Green for knowns

        # ==========================================
        # STEP 1: Draw EVERYTHING on the full frame
        # ==========================================
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = name
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_top_y = y1 - text_height - baseline - 10
        
        cv2.rectangle(frame, (x1, label_top_y), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ==========================================
        # STEP 2: Crop (after drawing)
        # ==========================================
        start_y = max(0, label_top_y - margin)
        end_y = min(height, y2 + margin)
        start_x = max(0, x1 - margin)
        
        max_x_needed = max(x2, x1 + text_width + 10)
        end_x = min(width, max_x_needed + margin)
        
        cropped_face = frame[start_y:end_y, start_x:end_x]
        
        # 3. Add to our dictionary for this frame
        faces_in_frame[name] = cropped_face
    
    return faces_in_frame



# if __name__ == "__main__":
#     main()
