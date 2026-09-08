import cv2
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

    def detect(self, image):
        """
        Detect faces in an image.

        Args:
            image: OpenCV BGR image

        Returns:
            List of detected faces
        """
        return self.app.get(image)
