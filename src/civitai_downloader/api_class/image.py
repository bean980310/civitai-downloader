from dataclasses import dataclass
from typing import Any, Dict, List

from civitai_downloader.api_class.metadata import Metadata
from civitai_downloader.api_class.common import NsfwLevel

@dataclass
class Image:
    id: int
    url: str
    hash: str
    width: int
    height: int
    nsfw: bool
    nsfwLevel: NsfwLevel
    createdAt: str
    postId: int
    stats: Dict[str, int]
    meta: Dict[str, Any]
    username: str

@dataclass
class ImageList:
    items: List[Image]
    metadata: Metadata