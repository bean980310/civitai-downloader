import requests
from civitai_downloader.api import CIVITAI_API_URL
from typing import Optional

def get_creators_info_from_api(limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None, include_metadata: Optional[bool]=False, api_token: Optional[str]=None):
    api_url=f'{CIVITAI_API_URL}/creators'
    headers={}
    headers['Authorization']=f'Bearer {api_token}' if api_token else None
    params={}
    params['limit']=limit
    params['page']=page
    params['query']=query
    response=requests.get(api_url, headers=headers, params=params)
    if response.status_code==200:
        data=response.json()
        creators_info=[]
        creators=data.get('items', [])
        for creator in creators:
            info={
                'username': creator.get('username'),
                'modelCount': creator.get('modelCount'),
                'link': creator.get('link')
            }
            creators_info.append(info)
        if include_metadata:
            metadata=data.get('metadata', {})
            creators_metadata={
                'totalItems': metadata.get('totalItems'),
                'currentPage': metadata.get('currentPage'),
                'pageSize': metadata.get('pageSize'),
                'totalPages': metadata.get('totalPages'),
                'nextPage': metadata.get('nextPage'),
                'prevPage': metadata.get('prevPage')
            }
            return creators_info, creators_metadata
        else:
            return creators_info
    else:
        print(f'{response.status_code} : {response.text}')
        return []