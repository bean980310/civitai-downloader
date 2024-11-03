from dataclasses import dataclass

from civitai_downloader.api_class.metadata import Metadata

@dataclass
class Creator:
    username: str
    modelCount: int
    link: str

@dataclass
class CreatorList:
    items: Creator
    metadata: Metadata