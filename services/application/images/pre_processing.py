import cv2
from ninja import UploadedFile

from services.application.images.pre_processing_base import ImagePreProcessingBase
from services.domain.image_processor import ImageProcessor
from services.domain.prompt import get_system_prompt, get_user_prompt
from services.domain.vaiv_caller import VaivCaller
from services.schema.receipt_output import ItemListModel


class ImagePreProcessing(ImagePreProcessingBase):
    JPEG_QUALITY = 95

    def extract_text(
        self, uploaded_files: list[UploadedFile], model: str
    ) -> tuple[str | ItemListModel | None, list[str]]:
        imgs = self._load_multi_images(uploaded_files)

        code = None
        provider = "ollama"
        b64s = []

        for img in imgs:
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

            b64s.append(self._to_base64(processed_image, code))

        result = VaivCaller(provider=provider).call_llm(
            get_system_prompt(), get_user_prompt(), b64s, model
        )

        return result, b64s
