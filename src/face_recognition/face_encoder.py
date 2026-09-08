import cv2
import numpy as np

from insightface.app import FaceAnalysis


class FaceEncoder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
# "DmlExecutionProvider",
        )
        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

    def encode(self, image):
        """
        Detect a face and generate its face embedding.

        Args:
            image: OpenCV BGR image

        Returns:
            numpy.ndarray: Face embedding

        Raises:
            ValueError: If no face or multiple faces are detected.
        """

        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError("No face detected in the image.")

        if len(faces) > 1:
            raise ValueError(
                f"Expected one face, but detected {len(faces)} faces."
            )

        face = faces[0]

        if face.embedding is None:
            raise ValueError("Could not generate face embedding.")

        embedding = np.asarray(face.embedding, dtype=np.float32)

        return embedding
