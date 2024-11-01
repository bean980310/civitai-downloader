import requests
from civitai_downloader.api import CIVITAI_API_URL
from typing import Optional

def get_images_info_from_api(limit: Optional[int]=100, 
                             post_id: Optional[int]=None,
                             model_id: Optional[int]=None,
                             model_version_id: Optional[int]=None,
                             username: Optional[str]=None,
                             nsfw: Optional[bool|enumerate]=None,
                             sort: Optional[enumerate]='Most Reactions',
                             period: Optional[enumerate]='Week',
                             page: Optional[int]=1,
                             api_token: Optional[str]=None):
    api_url=f'{CIVITAI_API_URL}/images'
    headers={}
    headers['Authorization']=f'Bearer {api_token}' if api_token else None
    params={}
    params['limit']=limit
    params['postId']=post_id
    params['modelId']=model_id
    params['modelVersionId']=model_version_id
    params['username']=username
    params['nsfw']=nsfw
    params['sort']=sort
    params['period']=period
    params['page']=page
    response=requests.get(api_url, headers=headers, params=params)
    if response.status_code==200:
        data=response.json()
        return data
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code