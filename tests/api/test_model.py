from civitai_downloader.api.model import get_model_info_simple_from_api, get_model_version_info_simple_from_api, get_model_info_from_api, get_model_version_info_from_api

def test_get_model_info_from_api():
    model_id=9409
    model_info=get_model_info_from_api(model_id)
    test_model_info = get_model_info_simple_from_api(model_id)
    model_name=model_info.get('name')
    model_type=model_info.get('type')
    model_poi=model_info.get('poi')
    model_is_nsfw=model_info.get('nsfw')
    model_allow_no_credit=model_info.get('allowNoCredit')
    model_allow_commercial_use=model_info.get('allowCommercialUse')
    model_creator_name=model_info.get('creator').get('username')
    assert test_model_info[0] == model_id
    assert test_model_info[1] == model_name
    assert test_model_info[2] == model_type
    assert test_model_info[3] == model_poi
    assert test_model_info[4] == model_is_nsfw
    assert test_model_info[5] == model_allow_no_credit
    assert test_model_info[6] == model_allow_commercial_use
    assert test_model_info[8] == model_creator_name

def test_get_model_version_info_from_api():
    model_version_id=9409
    model_version_info=get_model_version_info_from_api(model_version_id)
    test_model_version_info=get_model_version_info_simple_from_api(model_version_id)
    model_id=model_version_info.get('modelId')
    model_version_name=model_version_info.get('name')
    model_version_created=model_version_info.get('createdAt')
    model_version_updated=model_version_info.get('updatedAt')
    base_model=model_version_info.get('baseModel')
    model_version_desc=model_version_info.get('description')
    model_version_files_info=[]
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
    assert test_model_version_info[0] == model_version_id
    assert test_model_version_info[1] == model_id
    assert test_model_version_info[2] == model_version_name
    assert test_model_version_info[3] == model_version_created
    assert test_model_version_info[4] == model_version_updated
    assert test_model_version_info[6] == base_model
    assert test_model_version_info[7] == model_version_desc
    assert test_model_version_info[8] == model_version_files_info
    assert test_model_version_info[9] == model_version_images_url
    assert test_model_version_info[10] == model_version_download_url