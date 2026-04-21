import base64
from datetime import datetime

import cv2
import numpy as np
from ninja import UploadedFile

from services.domain.image_processor import ImageProcessor
from services.domain.prompt import get_system_prompt, get_user_prompt, get_vote_prompt
from services.domain.vaiv_caller import VaivCaller


class ImagePreProcessing:
    JPEG_QUALITY = 95

    @staticmethod
    def _load_image(uploaded_file: UploadedFile) -> np.ndarray:
        raw_bytes = uploaded_file.read()
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        return image

    def _to_base64(self, img: np.ndarray, code: int | None) -> str:
        if code:
            bgr = cv2.cvtColor(img, code)
        else:
            bgr = img

        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])

        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def extract_text(self, uploaded_file: UploadedFile, model: str) -> tuple[str | None, str]:
        img = self._load_image(uploaded_file)

        if model.startswith("gemma"):
            processed_image = (
                ImageProcessor(img=img)
                .to_grayscale_with_gamma(gamma=1.2)
                .denoise()
                .sharpen(alpha=1.7)
                # .enhance_contrast(clip_limit=2.0)
                # .adaptive_binarize(block_size=31)
                .get_image()
            )
            code = cv2.COLOR_GRAY2BGR
            provider = "vllm"
        else:
            processed_image = (
                ImageProcessor(img=img)
                .upscale(scale_factor=1.47)
                .rotate_image(angle=1.5)
                # .unsharp_mask()
                .get_image()
            )
            code = None
            provider = "ollama"

        b64 = self._to_base64(processed_image, code)

        result = VaivCaller(provider=provider).call_llm(
            get_system_prompt(), get_user_prompt(), b64, model
        )

        return result, b64

    async def extract_text_with_voting(
        self, uploaded_file: UploadedFile, model: str
    ) -> tuple[str | None, str]:
        img = self._load_image(uploaded_file)

        processed_images = []
        for angle in (0, 1.0, -1.0):
            image = ImageProcessor(img=img).upscale().rotate_image(angle=angle).get_image()
            processed_images.append(self._to_base64(image, None))

        print(f"{datetime.now()} - 이미지 전처리 끝 - LLM 호출")
        result = await VaivCaller(provider="ollama").vote(
            get_vote_prompt(), get_system_prompt(), get_user_prompt(), processed_images, model
        )

        return result, ""
