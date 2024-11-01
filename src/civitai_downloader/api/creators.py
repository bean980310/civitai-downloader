import requests
from civitai_downloader.api import CIVITAI_API_URL
from typing import Optional

def get_creators_info_from_api(limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None, api_token: Optional[str]=None):
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
        return data
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code
    
def get_creators_info_simple_from_api(limit: Optional[int]=20, page: Optional[int]=1, query: Optional[str]=None, api_token: Optional[str]=None):
    creators_db=get_creators_info_from_api(limit, page, query, api_token)
    if creators_db:
        creators_info=[]
        creators=creators_db.get('items', [])
        for creator in creators:
            info={
                'username': creator.get('username'),
                'modelCount': creator.get('modelCount'),
                'link': creator.get('link')
            }
            creators_info.append(info)
        return creators_info
    else:
        return None