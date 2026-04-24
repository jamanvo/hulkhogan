from ninja import Schema

from services.schema.receipt_output import ItemListModel


class OcrResponse(Schema):
    detail: ItemListModel
    enhanced_images: list[str] | None
