from typing import Optional
import requests
from civitai_downloader.api import CIVITAI_API_URL

def get_model_info_from_api(
        model_id: int, 
        api_token: Optional[str]=None, 
        include_desc: Optional[bool]=False
        ):
    api_url=f'{CIVITAI_API_URL}/models/{model_id}'
    headers={'Authorization': f'Bearer {api_token}'} if api_token is not None else {}
    response=requests.get(api_url, headers=headers)
    if response.status_code==200:
        data=response.json()
        model_name=data.get('name')
        model_desc=data.get('description') if include_desc else None
        model_type=data.get('type')
        model_is_nsfw=data.get('nsfw')
        model_tags=[data.get('tags')[i] for i in range(len(data.get('tags')))]
        model_mode=data.get('mode')
        model_creator_name=data.get('creator').get('username')
        model_version_id=[data.get('modelVersions')[i].get('id') for i in range(len(data.get('modelVersions')))]
        model_version_name=[data.get('modelVersions')[i].get('name') for i in range(len(data.get('modelVersions')))]
        model_version_url=[data.get('modelVersions')[i].get('downloadUrl') for i in range(len(data.get('modelVersions')))]
        return model_id, model_name, model_desc, model_type, model_is_nsfw, model_tags, model_mode, model_creator_name, model_version_id, model_version_name, model_version_url
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code
    
def get_model_version_info_from_api(
        model_version_id: int,
        api_token: Optional[str]=None, 
        include_desc: Optional[bool]=False
):
    api_url=f'{CIVITAI_API_URL}/model-versions/{model_version_id}'
    headers={'Authorization': f'Bearer {api_token}'} if api_token is not None else {}
    response=requests.get(api_url, headers=headers)
    if response.status_code==200:
        data=response.json()
        model_version_name=data.get('name')
        model_id=data.get('modelId')
        model_created=data.get('createdAt')
        model_updated=data.get('updatedAt')
        model_trained_words=[data.get('trainedWords')[i] for i in range(len(data.get('trainedWords')))]
        model_version_desc=data.get('description') if include_desc else None
        base_model=data.get('baseModel')
        return model_version_id, model_id, model_version_name, model_created, model_updated, model_trained_words, base_model, model_version_desc
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code