from typing import List

from ninja import File, Form, Router
from ninja.files import UploadedFile

from api.agent.schemas import OcrResponse
from services.application.images.nomotron_processing import NemotronPreProcessing
from services.application.images.pre_processing import ImagePreProcessing


router = Router(tags=["Agent"])


@router.post("/image-to-text/vaiv", response=OcrResponse)
def image_to_text(request, image_file: List[UploadedFile], model: Form[str]):
    answer, b64 = ImagePreProcessing().extract_text(image_file, model)

    return {"detail": answer, "enhanced_images": b64}


@router.post("/image-to-text/nemotron-ocr-v2", response=OcrResponse)
def image_to_text_nemotron(request, image_file: List[UploadedFile]):
    answer, b64s = NemotronPreProcessing().extract_text_with_ocr(image_file)

    return {"detail": answer, "enhanced_images": b64s}
