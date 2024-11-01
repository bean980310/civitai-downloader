from civitai_downloader.api.model import get_model_info_from_api, get_model_version_info_from_api

def test_get_model_info_from_api():
    model_info = get_model_info_from_api(9409)
    assert model_info[0] == 9409
    assert model_info[1] == '万象熔炉 | Anything XL'
    assert model_info[3] == "Checkpoint"
    assert model_info[4] == False
    assert model_info[5] == False
    assert model_info[6] == True
    assert model_info[7] == ['Image', 'RentCivit', 'Rent', 'Sell']
    assert model_info[9] == 'Yuno779'

def test_get_model_version_info_from_api():
    model_version_info=get_model_version_info_from_api(384264)
    assert model_version_info[0] == 384264
    assert model_version_info[1] == 9409
    assert model_version_info[2] == 'XL'
    assert model_version_info[3] == '2024-03-10T08:06:40.214Z'
    assert model_version_info[4] == '2024-09-19T04:27:26.481Z'
    assert model_version_info[6] == 'SDXL 1.0'
    assert model_version_info[8][0] == 'AnythingXL_xl.safetensors'
    assert model_version_info[9][0] == 'https://civitai.com/api/download/models/384264'
    assert model_version_info[10][0] == 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/4bdea703-131d-403c-89c7-d823ad8683a9/width=450/12622035.jpeg'
    assert model_version_info[11] == 'https://civitai.com/api/download/models/384264'