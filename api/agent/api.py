import base64

import anthropic
from anthropic.types import Base64ImageSourceParam, ImageBlockParam, MessageParam, TextBlockParam
from ninja import File, Form, Router
from ninja.files import UploadedFile

from services.application.images.pre_processing import ImagePreProcessing


router = Router(tags=["Agent"])


@router.post("/image-to-text/vaiv")
def image_to_text(request, image_file: File[UploadedFile], model: Form[str]):
    answer, b64 = ImagePreProcessing().extract_text(image_file, model)

    return {"detail": answer, "enhanced_image": b64}


@router.post("/image-to-text/vaiv-vote")
async def image_to_text_vote(request, image_file: File[UploadedFile], model: Form[str]):
    answer, b64 = await ImagePreProcessing().extract_text_with_voting(image_file, model)

    return {"detail": answer, "enhanced_image": b64}


@router.post("/image-to-text/anthropic")
def image_to_text_anthropic(request, image_file: UploadedFile):
    USER_PROMPT_TEMPLATE = """첨부한 이미지를 분석하여 내용을 알려주세요. 
        요약같은것은 하지 말고, 최대한 원본 내용을 알아야합니다. 
        이미지에 수기 메모 등 사람의 작업 흔적은 무시하세요."""

    b64 = base64.b64encode(image_file.read()).decode("utf-8")

    client = anthropic.Anthropic()

    messages = MessageParam(
        role="user",
        content=[
            ImageBlockParam(
                type="image",
                source=Base64ImageSourceParam(
                    data=b64,
                    type="base64",
                    media_type="image/jpeg",
                ),
            ),
            TextBlockParam(type="text", text=USER_PROMPT_TEMPLATE),
        ],
    )

    answer = client.messages.create(model="claude-sonnet-4-6", max_tokens=1000, messages=messages)

    return {"detail": answer.content[0].text, "enhanced_image": b64}
