from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List

from civitai_downloader.api_class.common import BaseModel, ModelType, AllowCommercialUse, ModelMode
from civitai_downloader.api_class.model_version import ModelVersionFile, ModelVersionImages

# Response Fields Classes
@dataclass
class ModelVersions:
    id: int
    modelId: int
    name: str
    createdAt: date
    updatedAt: date
    trainedWords: List[str]
    baseModel: BaseModel
    earlyAccessTimeFrame: int
    description: str
    stats: Dict[str, Any]
    files: List[ModelVersionFile]
    images: List[ModelVersionImages]
    downloadUrl: str

@dataclass
class Model:
    id: int
    name: str
    description: str
    type: ModelType
    poi: bool
    nsfw: bool
    allowNoCredit: bool
    allowCommercialUse: AllowCommercialUse
    allowDerivates: bool
    allowDifferentLicense: bool
    stats: Dict[str, Any]
    creator: Dict[str, str]
    tags: List[str]
    modelVersions: List[ModelVersions]
    mode: ModelMode