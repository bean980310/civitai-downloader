from civitai_downloader.token.token import get_token, prompt_for_civitai_token

def test_get_token():
    token = get_token()
    assert token is not None
    assert isinstance(token, str)