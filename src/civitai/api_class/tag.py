from dataclasses import dataclass
from civitai.api_class.metadata import Metadata

@dataclass
class Tag:
    name: str
    modelCount: int
    link: str

@dataclass
class TagList:
    items: Tag
    metadata: Metadata