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
        model_desc=data.get('description') if include_desc else ''
        model_type=data.get('type')
        model_poi=data.get('poi')
        model_is_nsfw=data.get('nsfw')
        model_allow_no_credit=data.get('allowNoCredit')
        model_allow_commercial_use=data.get('allowCommercialUse')
        model_stats={
            'downloadCount': data.get('stats').get('downloadCount'),
            'favoriteCount': data.get('stats').get('favoriteCount'),
            'commentCount': data.get('stats').get('commentCount'),
            'ratingCount': data.get('stats').get('ratingCount'),
            'rating': data.get('stats').get('rating')
        }
        model_creator_name=data.get('creator').get('username')
        model_creator_image=data.get('creator').get('image')
        model_tags=[data.get('tags')[i] for i in range(len(data.get('tags')))]
        model_version=[{
            'id': data.get('modelVersions')[i].get('id'),
            'name': data.get('modelVersions')[i].get('name'),
            'createdAt': data.get('modelVersions')[i].get('createdAt'),
            'updatedAt': data.get('modelVersions')[i].get('updatedAt'),
            'trainedWords': [data.get('modelVersions')[i].get('trainedWords')[j] for j in range(len(data.get('modelVersions')[i].get('trainedWords')))],
            'baseModel': data.get('modelVersions')[i].get('baseModel'),
            'description': data.get('modelVersions')[i].get('description') if include_desc else '',
            'downloadUrl': data.get('modelVersions')[i].get('downloadUrl')
        } for i in range(len(data.get('modelVersions')))]
        return model_id, model_name, model_desc, model_type, model_poi, model_is_nsfw, model_allow_no_credit, model_allow_commercial_use, model_stats, model_creator_name, model_creator_image, model_tags, model_version
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