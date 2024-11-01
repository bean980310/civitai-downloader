from typing import Optional
import requests
from civitai_downloader.api import CIVITAI_API_URL

def get_nodels_info_from_api(
        limit: Optional[int]=100,
        page: Optional[int]=1,
        query: Optional[str]=None,
        tag: Optional[str]=None,
        username: Optional[str]=None,
        types: Optional[enumerate]=None,
        sort: Optional[enumerate]='Highest Rated',
        period: Optional[enumerate]='Week',
        primary_file_only: Optional[bool]=False,
        allow_no_credit: Optional[bool]=False,
        allow_derivates: Optional[bool]=False,
        allow_different_licenses: Optional[bool]=False,
        allow_commercial_use: Optional[enumerate]=None,
        nsfw: Optional[bool]=False,
        supports_generation: Optional[bool]=False,
        api_token: Optional[str]=None
):
    api_url=f'{CIVITAI_API_URL}/models'
    headers={}
    headers['Authorization']=f'Bearer {api_token}' if api_token else None
    params={}
    params['limit']=limit
    params['page']=page
    params['query']=query
    params['tag']=tag
    params['username']=username
    params['types']=types
    params['sort']=sort
    params['period']=period
    params['primaryFileOnly']=primary_file_only
    params['allowNoCredit']=allow_no_credit
    params['allowDerivates']=allow_derivates
    params['allowDifferentLicenses']=allow_different_licenses
    params['allowCommercialUse']=allow_commercial_use
    params['nsfw']=nsfw
    params['supportsGeneration']=supports_generation
    response=requests.get(api_url, headers=headers, params=params)
    if response.status_code==200:
        data=response.json()
        return data
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code