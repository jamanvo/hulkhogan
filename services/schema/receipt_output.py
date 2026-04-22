from pydantic import BaseModel, Field


class ItemModel(BaseModel):
    name: str = Field(..., description="품목명")
    count: int = Field(..., description="수량/무게/중량/부피. 반드시 기재")
    price: int = Field(..., description="결제금액|가격|금액")
    unit_price: int = Field(None, description="단가: 1단위당 가격")
    unit: str | None = Field(None, description="단위: 개, kg, g. 수량은 기재하지 않음")


class ItemListModel(BaseModel):
    items: list[ItemModel] = Field(..., description="품목 목록")
