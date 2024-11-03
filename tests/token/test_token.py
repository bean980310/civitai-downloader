from civitai_downloader.token.token import get_token, prompt_for_civitai_token

def test_token():
    token=get_token()
    if token is None:
        token=prompt_for_civitai_token()
    assert token