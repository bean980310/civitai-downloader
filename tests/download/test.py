from civitai_downloader.download.download import civitai_download, url_download
from civitai_downloader.token.token import get_token, prompt_for_civitai_token

token=get_token()
if token is None:
    token=prompt_for_civitai_token()

def test_civitai_download():
    model_id=215522
    path="."
    civitai_dl=civitai_download(model_id=model_id, local_dir=path, token=token)
    assert civitai_dl is not None
    assert isinstance(civitai_dl, tuple)
    assert len(civitai_dl)==3
    assert isinstance(civitai_dl[0], str)
    assert isinstance(civitai_dl[1], str)
    assert isinstance(civitai_dl[2], str)

def test_url_download():
    url="https://civitai.com/api/download/models/215522"
    path="."
    url_dl=url_download(url=url, local_dir=path, token=token)
    assert url_dl is not None
    assert isinstance(url_dl, tuple)
    assert len(url_dl)==3
    assert isinstance(url_dl[0], str)
    assert isinstance(url_dl[1], str)
    assert isinstance(url_dl[2], str)
