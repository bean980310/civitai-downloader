from typing import Optional, Dict

from civitai_downloader.api.base import BaseAPI, CIVITAI_API_URL
from civitai_downloader.api.creators import CreatorsAPI
from civitai_downloader.api.images import ImagesAPI
from civitai_downloader.api.model_version import ModelVersionAPI
from civitai_downloader.api.model import ModelAPI
from civitai_downloader.api.models import ModelsAPI
from civitai_downloader.api.tags import TagsAPI
from civitai_downloader.api_class import (
    CreatorList, ImageList, Model, ModelList, ModelVersion, TagList,
    Sort, Period, ModelType, AllowCommercialUse, BaseModel, NsfwLevel
)

class CivitAIClient(BaseAPI):
    """
    Unified client interface for the CivitAI API.
    Manages all API endpoints through a single interface while maintaining
    the independence of individual API classes.
    
    Inherits from BaseAPI to maintain consistent authentication and base URL handling.
    """
    def __init__(self, api_token: Optional[str] = None, api_url: str = CIVITAI_API_URL):
        super().__init__(api_token)
        self.api_url = api_url
        
        # Initialize all API handlers
        self._creators = CreatorsAPI(api_token=api_token, api_url=api_url)
        self._images = ImagesAPI(api_token=api_token, api_url=api_url)
        self._model_version = ModelVersionAPI(api_token=api_token, api_url=api_url)
        self._model = ModelAPI(api_token=api_token, api_url=api_url)
        self._models = ModelsAPI(api_token=api_token, api_url=api_url)
        self._tags = TagsAPI(api_token=api_token, api_url=api_url)

    # Base methods
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        return self._get_headers()

    # Creators endpoints
    def list_creators(self, limit: Optional[int] = 20, page: Optional[int] = 1, 
                     query: Optional[str] = None) -> CreatorList:
        return self._creators.list_creators(limit, page, query)

    # Images endpoints
    def list_images(self, limit: Optional[int] = 100, postId: Optional[int] = None,
                   modelId: Optional[int] = None, modelVersionId: Optional[int] = None,
                   username: Optional[str] = None, nsfw: Optional[NsfwLevel] = None,
                   sort: Optional[Sort] = None, period: Optional[Period] = None,
                   page: Optional[int] = 1) -> ImageList:
        return self._images.list_images(
            limit, postId, modelId, modelVersionId, username,
            nsfw, sort, period, page
        )

    # Model Version endpoints
    def get_model_version(self, model_version_id: int) -> ModelVersion:
        return self._model_version.get_model_version_info_from_api(model_version_id)

    def get_model_version_by_hash(self, hash: str) -> ModelVersion:
        return self._model_version.get_model_version_info_by_hash_from_api(hash)

    # Models endpoints
    def get_model(self, model_id: int) -> Model:
        return self._model.get_model_info_from_api(model_id)

    def list_models(self, limit: Optional[int] = 100, page: Optional[int] = 1,
                   query: Optional[str] = None, tag: Optional[str] = None,
                   username: Optional[str] = None, types: Optional[ModelType] = None,
                   sort: Optional[Sort] = "Highest Rated", period: Optional[Period] = "Week",
                   favorites: Optional[bool] = None, hidden: Optional[bool] = None,
                   primaryFileOnly: Optional[bool] = None, allowNoCredit: Optional[bool] = None,
                   allowDerivates: Optional[bool] = None, allowDifferentLicenses: Optional[bool] = None,
                   allowCommercialUse: Optional[AllowCommercialUse] = None,
                   baseModel: Optional[BaseModel] = None, nsfw: Optional[bool] = None,
                   supportsGeneration: Optional[bool] = None) -> ModelList:
        return self._models.list_models(
            limit, page, query, tag, username, types, sort, period,
            favorites, hidden, primaryFileOnly, allowNoCredit,
            allowDerivates, allowDifferentLicenses, allowCommercialUse,
            baseModel, nsfw, supportsGeneration
        )

    # Tags endpoints
    def list_tags(self, limit: Optional[int] = 20, page: Optional[int] = 1,
                 query: Optional[str] = None) -> TagList:
        return self._tags.list_tags(limit, page, query)