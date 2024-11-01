from typing import Optional
import requests
from civitai_downloader.api import CIVITAI_API_URL

def get_tags_info_from_api(limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None, api_token: Optional[str]=None):
    api_url=f'{CIVITAI_API_URL}/tags'
    headers={}
    headers['Authorization']=f'Bearer {api_token}' if api_token else None
    params={}
    params['limit']=limit
    params['page']=page
    params['query']=query
    response=requests.get(api_url, headers=headers, params=params)
    if response.status_code==200:
        data=response.json()
        return data
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code
    
def get_tags_info_simple_from_api(limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None, api_token: Optional[str]=None):
    tags_db=get_tags_info_from_api(limit, page, query, api_token)
    if tags_db:
        tags_info=[]
        tags=tags_db.get('items', [])
        for tag in tags:
            info={
                'name': tag.get('name'),
                'modelCount': tag.get('modelCount'),
                'link': tag.get('link')
            }
            tags_info.append(info)
        return tags_info
    else:
        return None