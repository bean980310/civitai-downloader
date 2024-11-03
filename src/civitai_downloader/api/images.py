import requests
from typing import Optional, Dict

from civitai_downloader.api.base import BaseAPI
from civitai_downloader.api_class import Image, ImageList, NsfwLevel, Sort, Period, Metadata

class ImagesAPI(BaseAPI):
    def list_images(self, 
                    limit: Optional[int]=100,
                    postId: Optional[int]=None,
                    modelId: Optional[int]=None,
                    modelVersionId: Optional[int]=None,
                    username: Optional[str]=None,
                    nsfw: Optional[NsfwLevel]=None,
                    sort: Optional[Sort]=None,
                    period: Optional[Period]=None,
                    page: Optional[int]=1)->ImageList:
        api_url=f'{self.api_url}/images'
        headers=self._get_headers()
        params = self._construct_params(locals())

        response=requests.get(api_url, params=params, headers=headers)
        if response.status_code==200:
            data=response.json()
            images=[]
            for image_data in data.get('items'):
                image=self._parse_image(image_data)
                images.append(image)

            metadata_data=data.get('metadata', {})
            metadata=Metadata(
                totalItems=metadata_data.get('totalItems', 0),
                currentPage=metadata_data.get('currentPage', 1),
                pageSize=metadata_data.get('pageSize', len(images)),
                totalPages=metadata_data.get('totalPages', 1),
                nextPage=metadata_data.get('nextPage'),
                prevPage=metadata_data.get('prevPage')
            )
            image_list=ImageList(items=images, metadata=metadata)
            return image_list

    def _construct_params(self, kwargs):
        params={
            'limit': kwargs.get('limit'),
            'postId': kwargs.get('postId'),
            'modelId': kwargs.get('modelId'),
            'modelVersionId': kwargs.get('modelVersionId'),
            'username': kwargs.get('username'),
            'nsfw': kwargs.get('nsfw').value if kwargs.get('nsfw') else None,
            'sort': kwargs.get('sort').value if kwargs.get('sort') else None,
            'period': kwargs.get('period').value if kwargs.get('period') else None,
            'page': kwargs.get('page')
        }
        return {k: v for k, v in params.items() if v is not None}
    
    def _parse_image(self, data: Dict)->Image:
        image=Image(
            id=data.get('id'),
            url=data.get('url'),
            hash=data.get('hash'),
            width=data.get('width'),
            height=data.get('height'),
            nsfw=data.get('nsfw'),
            nsfwLevel=data.get('nsfwLevel'),
            createdAt=data.get('createdAt'),
            postId=data.get('postId'),
            stats=data.get('stats', {}),
            meta=data.get('meta', {}),
            username=data.get('username')
        )
        return image
