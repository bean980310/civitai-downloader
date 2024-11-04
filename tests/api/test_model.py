from civitai_downloader.api.model import ModelAPI
from civitai_downloader.api.model_version import ModelVersionAPI
from civitai_downloader.token.token import get_token

token=get_token()

def test_get_model_info_from_api():
    api=ModelAPI(api_token=token)
    model_id=9409
    model=api.get_model_info_from_api(model_id=model_id)
    model_name=model.name
    model_type=model.type
    model_poi=model.poi
    model_is_nsfw=model.nsfw
    model_allow_no_credit=model.allowNoCredit
    model_allow_commercial_use=model.allowCommercialUse
    model_creator=model.creator

    assert model.id == model_id
    assert model.name == model_name
    assert model.type == model_type
    assert model.poi == model_poi
    assert model.nsfw == model_is_nsfw
    assert model.allowNoCredit == model_allow_no_credit
    assert model.allowCommercialUse == model_allow_commercial_use
    assert model.creator == model_creator

def test_get_model_version_info_from_api():
    api=ModelVersionAPI(api_token=token)
    model_version_id=9409
    version=api.get_model_version_info_from_api(model_version_id)
    model_id=version.modelId
    model_version_name=version.name
    model_version_created=version.createdAt
    model_version_updated=version.updatedAt
    base_model=version.baseModel
    model_version_desc=version.description

    assert version.id == model_version_id
    assert version.modelId == model_id
    assert version.name == model_version_name
    assert version.createdAt == model_version_created
    assert version.updatedAt == model_version_updated
    assert version.baseModel == base_model
    assert version.description == model_version_desc