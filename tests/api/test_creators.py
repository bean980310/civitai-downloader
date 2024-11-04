from civitai_downloader.api.creators import CreatorsAPI
from civitai_downloader.token.token import get_token

token=get_token()

def test_get_creators_info_simple_from_api():
    api=CreatorsAPI(api_token=token)
    creators = api.list_creators()
    assert creators