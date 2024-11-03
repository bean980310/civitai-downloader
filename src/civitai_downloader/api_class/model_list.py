from dataclasses import dataclass
from typing import List

from civitai_downloader.api_class.metadata import Metadata
from civitai_downloader.api_class.model import Model

@dataclass
class ModelList:
    items: List[Model]
    metadata: Metadata