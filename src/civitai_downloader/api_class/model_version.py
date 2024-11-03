from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List

from civitai_downloader.api_class.common import ModelFp, ModelSize, ModelFormat, BaseModel


@dataclass
class ModelVersionFileMetadata:
    fp: ModelFp
    size: ModelSize
    format: ModelFormat

@dataclass
class ModelVersionFile:
    name: str
    id: int
    sizeKB: float
    type: str
    metadata: List[ModelVersionFileMetadata]
    pickleScanResult: str
    pickleScanMessage: str
    virusScanResult: str
    scannedAt: str
    hashes: Dict[str, str]
    primary: bool
    downloadUrl: str

@dataclass
class ModelVersionImages:
    url: str
    nsfw: str
    width: int
    height: int
    hash: str
    meta: Dict[str, Any]

@dataclass
class ModelVersion:
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
    model: Dict[str, Any]
    files: List[ModelVersionFile]
    images: List[ModelVersionImages]
    downloadUrl: str
    
    