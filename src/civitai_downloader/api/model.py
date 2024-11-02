from typing import Optional
import requests
from civitai_downloader.api import CIVITAI_API_URL

def get_model_info_from_api(
        model_id: int, 
        api_token: Optional[str]=None
        ):
    api_url=f'{CIVITAI_API_URL}/models/{model_id}'
    headers={'Authorization': f'Bearer {api_token}'} if api_token is not None else {}
    response=requests.get(api_url, headers=headers)
    if response.status_code==200:
        data=response.json()
        return data
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code
    
def get_model_info_simple_from_api(
        model_id: int, 
        api_token: Optional[str]=None, 
        ):
    model_info=get_model_info_from_api(model_id, api_token)
    if model_id:
        model_name=model_info.get('name')
        model_type=model_info.get('type')
        model_poi=model_info.get('poi')
        model_is_nsfw=model_info.get('nsfw')
        model_allow_no_credit=model_info.get('allowNoCredit')
        model_allow_commercial_use=model_info.get('allowCommercialUse')
        model_stats={
            'downloadCount': model_info.get('stats').get('downloadCount'),
            'favoriteCount': model_info.get('stats').get('favoriteCount'),
            'commentCount': model_info.get('stats').get('commentCount'),
            'ratingCount': model_info.get('stats').get('ratingCount'),
            'rating': model_info.get('stats').get('rating')
        }
        model_creator_name=model_info.get('creator').get('username')
        model_creator_image=model_info.get('creator').get('image')
        model_tags=model_info.get('tags')
        model_version_info=[]
        model_versions=model_info.get('modelVersions', [])
        for model_version in model_versions:
            info={
                'id': model_version.get('id'),
                'name': model_version.get('name'),
                'createdAt': model_version.get('createdAt'),
                'updatedAt': model_version.get('updatedAt'),
                'trainedWords': model_version.get('trainedWords'),
                'baseModel': model_version.get('baseModel'),
                'description': model_version.get('description'),
                'downloadUrl': model_version.get('downloadUrl'),
                'files.name': [model_version.get('files')[i].get('name') for i in range(len(model_version.get('files')))],
                'files.downloadUrl': [model_version.get('files')[i].get('downloadUrl') for i in range(len(model_version.get('files')))],
                'images.url': [model_version.get('images')[i].get('url') for i in range(len(model_version.get('images')))]
            }
            model_version_info.append(info)
        return model_id, model_name, model_type, model_poi, model_is_nsfw, model_allow_no_credit, model_allow_commercial_use, model_stats, model_creator_name, model_creator_image, model_tags, model_version_info
    else:
        return None
    
def get_model_version_info_from_api(
        model_version_id: int,
        api_token: Optional[str]=None, 
):
    api_url=f'{CIVITAI_API_URL}/model-versions/{model_version_id}'
    headers={'Authorization': f'Bearer {api_token}'} if api_token is not None else {}
    response=requests.get(api_url, headers=headers)
    if response.status_code==200:
        data=response.json()
        return data
    
def get_model_version_info_simple_from_api(
        model_version_id: int,
        api_token: Optional[str]=None, 
):
    model_version_info=get_model_version_info_from_api(model_version_id, api_token)
    if model_version_id:
        model_version_name=model_version_info.get('name')
        model_id=model_version_info.get('modelId')
        model_created=model_version_info.get('createdAt')
        model_updated=model_version_info.get('updatedAt')
        model_trained_words=[model_version_info.get('trainedWords')[i] for i in range(len(model_version_info.get('trainedWords')))]
        base_model=model_version_info.get('baseModel')
        model_version_desc=model_version_info.get('description')
        model_version_files_info=[]
        model_version_images_info=[]
        files=model_version_info.get('files', [])
        for file in files:
            info={
                'name': file.get('name'),
                'downloadUrl': file.get('downloadUrl'),
                'type':file.get('type'),
                'metadata':file.get('metadata')
            }
            model_version_files_info.append(info)
        model_version_images_url=[model_version_info.get('images')[i].get('url') for i in range(len(model_version_info.get('images')))]
        model_version_download_url=model_version_info.get('downloadUrl')
        return model_version_id, model_id, model_version_name, model_created, model_updated, model_trained_words, base_model, model_version_desc, model_version_files_info, model_version_images_url, model_version_download_url
    else:
        return None
    
def get_model_version_info_by_hash_from_api(
        model_hash: str,
        api_token: Optional[str]=None
):
    api_url=f'{CIVITAI_API_URL}/model-versions/by-hash/{model_hash}'
    headers={'Authorization': f'Bearer {api_token}'} if api_token is not None else {}
    response=requests.get(api_url, headers=headers)
    if response.status_code==200:
        data=response.json()
        return data
    
def get_model_version_info_simple_by_hash_from_api(
        model_hash: str,
        api_token: Optional[str]=None
):
    model_version_info=get_model_version_info_by_hash_from_api(model_hash, api_token)
    if model_hash:
        model_version_id=model_version_info.get('id')
        model_version_name=model_version_info.get('name')
        model_id=model_version_info.get('modelId')
        model_created=model_version_info.get('createdAt')
        model_updated=model_version_info.get('updatedAt')
        model_trained_words=[model_version_info.get('trainedWords')[i] for i in range(len(model_version_info.get('trainedWords')))]
        base_model=model_version_info.get('baseModel')
        model_version_desc=model_version_info.get('description')
        model_version_files_info=[]
        model_version_images_info=[]
        files=model_version_info.get('files', [])
        for file in files:
            info={
                'name': file.get('name'),
                'downloadUrl': file.get('downloadUrl'),
                'type':file.get('type'),
                'metadata':file.get('metadata')
            }
            model_version_files_info.append(info)
        model_version_images_url=[model_version_info.get('images')[i].get('url') for i in range(len(model_version_info.get('images')))]
        model_version_download_url=model_version_info.get('downloadUrl')
        return model_version_id, model_id, model_version_name, model_created, model_updated, model_trained_words, base_model, model_version_desc, model_version_files_info, model_version_images_url, model_version_download_url
    else:
        return None