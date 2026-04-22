import base64
import os

import cv2
import numpy as np
from ninja import UploadedFile


class ImagePreProcessingBase:
    JPEG_QUALITY = 95

    @staticmethod
    def _load_image(uploaded_file: UploadedFile) -> np.ndarray:
        raw_bytes = uploaded_file.read()
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        return image

    @staticmethod
    def _load_multi_images(uploaded_files: list[UploadedFile]) -> list[np.ndarray]:
        result = []

        for uploaded_file in uploaded_files:
            raw_bytes = uploaded_file.read()
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            result.append(image)

        return result

    def _to_base64(self, img: np.ndarray, code: int | None) -> str:
        if code:
            bgr = cv2.cvtColor(img, code)
        else:
            bgr = img

        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])

        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _save_image(img: np.ndarray, image_path: str) -> None:
        cv2.imwrite(image_path, img)

    @staticmethod
    def _delete_file(file_path: str) -> None:
        os.remove(file_path)
