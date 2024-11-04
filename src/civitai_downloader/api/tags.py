from typing import Optional, Dict
import requests

from civitai_downloader.api.base import BaseAPI
from civitai_downloader.api_class import Tag, TagList, Metadata

class TagsAPI(BaseAPI):
    def list_tags(self, limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None)->TagList:
        api_url=f'{self.api_url}/tags'
        headers=self._get_headers()
        params = self._construct_params(locals())
        response=requests.get(api_url, headers=headers, params=params)
        if response.status_code==200:
            data=response.json()
            tags=[]
            for tag_data in data.get('items', []):
                tag=self._parse_tag(tag_data)
                tags.append(tag)
            metadata_data=data.get('metadata', {})
            metadata=Metadata(
                totalItems=metadata_data.get('totalItems', 0),
                currentPage=metadata_data.get('currentPage', 1),
                pageSize=metadata_data.get('pageSize', len(tags)),
                totalPages=metadata_data.get('totalPages', 1),
                nextPage=metadata_data.get('nextPage'),
                prevPage=metadata_data.get('prevPage')
            )
            tags_list=TagList(items=tags, metadata=metadata)
            return tags_list
        else:
            response.raise_for_status()

    def _construct_params(self, kwargs):
        params={
            'limit': kwargs.get('limit'),
            'page': kwargs.get('page'),
            'query': kwargs.get('query')
        }
        return {k: v for k, v in params.items() if v is not None}
    
    def _parse_tag(self, data: Dict)->Tag:
        tag=Tag(
            username=data.get('username'),
            modelCount=data.get('modelCount'),
            link=data.get('link')
        )
        return tag