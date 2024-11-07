from enum import Enum
from typing import Any, Protocol, TypeGuard, List, Optional, Union  # TypedDict
from typing_extensions import TypedDict
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, RootModel, Field
from pandas import DataFrame
import cv2
import base64

class StrEnum(str, Enum):
    value: str

    def __str__(self) -> str:
        return self.value

class Rect(dict):
    def __init__(self, x, y, w, h):
        super().__init__(x=x, y=y, width=w, height=h)

        self.x = x
        self.y = y
        self.w = w
        self.h = h


class TextResponse(BaseModel):
    try:
        name: str
    except:
        __root__: str


class MessageResponse(BaseModel):
    message: str



class BoundingBox(TypedDict): # TypedDict
    x1: int
    y1: int
    x2: int
    y2: int

    def to_xywh(self):
        x = self["x1"]
        y = self["y1"]
        width = self["x2"] - self["x1"]
        height = self["y2"] - self["y1"]
        return {"x": x, "y": y, "width": width, "height": height}

class AddManyResponse(BaseModel):
    ids: List[str]
    failed_indexes: List[str]

class GetResponse(BaseModel):
    name: Optional[str] = None
    id: Optional[str] = None
    boundingBox: Optional[BoundingBox] = None
    distance: Optional[float] = None
    embedding: Optional[Union[List[float],  List[np.ndarray]]] = None
    # img: Optional[np.ndarray] = None
    img: Optional[Union[str, np.ndarray]] = Field(default=None, example=None, description='List of base64 encoded images')
    class Config:
        arbitrary_types_allowed = True
    @staticmethod
    def array_to_base64(image_array: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_array)  # Chuyển đổi ảnh thành định dạng JPEG
        image_base64 = base64.b64encode(buffer).decode('utf-8')  # Mã hóa ảnh thành base64 và chuyển đổi thành chuỗi
        return image_base64

    @staticmethod
    def base64_to_array(image_base64: str) -> np.ndarray:
        image_data = base64.b64decode(image_base64)  # Giải mã dữ liệu base64
        image_array = np.frombuffer(image_data, dtype=np.uint8)  # Chuyển đổi dữ liệu thành numpy array
        image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Giải mã ảnh
        return image_array

    def bounding_box_to_xywh(self):
        return self.boundingBox.to_xywh()
    

class SearchResponse(BaseModel):
    results: List[GetResponse]

class QueryResponse(BaseModel):
    results: List[GetResponse]

class ModelType(StrEnum):
    CLIP = "clip"
    FACIAL_RECOGNITION = "facial-recognition"


class ModelRuntime(StrEnum):
    ONNX = "onnx"
    ARMNN = "armnn"


class HasProfiling(Protocol):
    profiling: dict[str, float]


class Face(TypedDict):
    boundingBox: BoundingBox
    embedding: npt.NDArray[np.float32]
    imageWidth: int
    imageHeight: int
    score: float

class DataFrameResponse(BaseModel):
    data: List[List[Union[str, float, int, None]]]  # Dữ liệu của DataFrame, mỗi dòng là một danh sách các giá trị
    columns: List[str]     # Tên các cột của DataFrame
    def to_dataframe(self) -> DataFrame:
        return DataFrame(self.data, columns=self.columns)
    
    
def has_profiling(obj: Any) -> TypeGuard[HasProfiling]:
    return hasattr(obj, "profiling") and isinstance(obj.profiling, dict)


def is_ndarray(obj: Any, dtype: "type[np._DTypeScalar_co]") -> "TypeGuard[npt.NDArray[np._DTypeScalar_co]]":
    return isinstance(obj, np.ndarray) and obj.dtype == dtype
