import base64
from typing import List

import anthropic
from anthropic.types import Base64ImageSourceParam, ImageBlockParam, MessageParam, TextBlockParam
from ninja import File, Form, Router
from ninja.files import UploadedFile

from api.agent.schemas import OcrResponse
from services.application.images.nomotron_processing import NemotronPreProcessing
from services.application.images.pre_processing import ImagePreProcessing
from services.domain.prompt import get_anthropic_system_prompt, get_system_prompt, get_user_prompt
from services.schema.receipt_output import ItemListModel


router = Router(tags=["Agent"])


@router.post("/image-to-text/vaiv", response=OcrResponse)
def image_to_text(request, image_file: List[UploadedFile], model: Form[str]):
    answer, b64 = ImagePreProcessing().extract_text(image_file, model)

    return {"detail": answer, "enhanced_images": b64}


@router.post("/image-to-text/nemotron-ocr-v2", response=OcrResponse)
def image_to_text_nemotron(request, image_file: List[UploadedFile]):
    answer, b64s = NemotronPreProcessing().extract_text_with_ocr(image_file)

    return {"detail": answer, "enhanced_images": b64s}


@router.post("/image-to-text/anthropic", response=OcrResponse)
def image_to_text_anthropic(request, image_file: List[UploadedFile]):
    b64s = [base64.b64encode(img.read()).decode("utf-8") for img in image_file]

    client = anthropic.Anthropic()
    img_block = [
        ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                data=b64,
                type="base64",
                media_type="image/jpeg",
            ),
        )
        for b64 in b64s
    ]

    messages = [
        MessageParam(
            role="assistant",
            content=[TextBlockParam(type="text", text=get_anthropic_system_prompt())],
        ),
        MessageParam(
            role="user",
            content=[
                *img_block,
                TextBlockParam(type="text", text=get_user_prompt()),
            ],
        ),
    ]

    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=messages,
        output_format=ItemListModel,
        temperature=0,
    )

    return {"detail": response.parsed_output, "enhanced_images": b64s}
