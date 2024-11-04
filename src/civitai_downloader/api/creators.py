import requests
from typing import Optional
from typing import Dict

from civitai_downloader.api.base import BaseAPI
from civitai_downloader.api_class import Creator, CreatorList, Metadata

class CreatorsAPI(BaseAPI):
    def list_creators(self, limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None)->CreatorList:
        api_url=f'{self.api_url}/creators'
        headers=self._get_headers()
        params = self._construct_params(locals())
        response=requests.get(api_url, headers=headers, params=params)
        if response.status_code==200:
            data=response.json()
            creators=[]
            for creator_data in data.get('items', []):
                creator=self._parse_creator(creator_data)
                creators.append(creator)
            metadata_data=data.get('metadata', {})
            metadata=Metadata(
                totalItems=metadata_data.get('totalItems', 0),
                currentPage=metadata_data.get('currentPage', 1),
                pageSize=metadata_data.get('pageSize', len(creators)),
                totalPages=metadata_data.get('totalPages', 1),
                nextPage=metadata_data.get('nextPage'),
                prevPage=metadata_data.get('prevPage')
            )
            creators_list=CreatorList(items=creators, metadata=metadata)
            return creators_list
        else:
            response.raise_for_status()

    def _construct_params(self, kwargs):
        params={
            'limit': kwargs.get('limit'),
            'page': kwargs.get('page'),
            'query': kwargs.get('query')
        }
        return {k: v for k, v in params.items() if v is not None}
    
    def _parse_creator(self, data: Dict)->Creator:
        creator=Creator(
            username=data.get('username'),
            modelCount=data.get('modelCount'),
            link=data.get('link')
        )
        return creator