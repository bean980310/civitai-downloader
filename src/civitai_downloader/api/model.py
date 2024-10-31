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
        model_version=data.get('modelVersions', {})
        model_version_info=[{
            'id': model_version[i].get('id'),
            'name': model_version[i].get('name'),
            'createdAt': model_version[i].get('createdAt'),
            'updatedAt': model_version[i].get('updatedAt'),
            'trainedWords': model_version[i].get('trainedWords'),
            'baseModel': model_version[i].get('baseModel'),
            'description': model_version[i].get('description'),
            'downloadUrl': model_version[i].get('downloadUrl'),
            'files.name': [model_version[i].get('files')[j].get('name') for j in range(len(model_version[i].get('files')))],
            'images.url': [model_version[i].get('images')[j].get('url') for j in range(len(model_version[i].get('images')))]
        } for i in range(len(model_version))]
        return model_id, model_name, model_desc, model_type, model_poi, model_is_nsfw, model_allow_no_credit, model_allow_commercial_use, model_stats, model_creator_name, model_creator_image, model_tags, model_version_info
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code
    
def get_model_version_info_from_api(
        model_version_id: int,
        api_token: Optional[str]=None, 
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
        base_model=data.get('baseModel')
        model_version_desc=data.get('description')
        model_version_files_name=[data.get('files')[i].get('name') for i in range(len(data.get('files')))]
        model_version_files_url=[data.get('files')[i].get('downloadUrl') for i in range(len(data.get('files')))]
        model_version_images_url=[data.get('images')[i].get('url') for i in range(len(data.get('images')))]
        model_version_download_url=data.get('downloadUrl')
        return model_version_id, model_id, model_version_name, model_created, model_updated, model_trained_words, base_model, model_version_desc, model_version_files_name, model_version_files_url, model_version_images_url, model_version_download_url
    else:
        error_code=f'{response.status_code} : {response.text}'
        return error_code