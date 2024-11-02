from civitai_downloader.download.download import civitai_download, advanced_download
from civitai_downloader.token.token import get_token, prompt_for_civitai_token

token=get_token()
if token is None:
    token=prompt_for_civitai_token()

def test_civitai_download():
    model_id=215522
    path="."
    url=f'https://civitai.com/api/download/models/{model_id}'
    civitai_dl=civitai_download(model_id=model_id, local_dir=path, token=token)
    assert url==civitai_dl[0]
    assert path==civitai_dl[1]

def test_advanced_download():
    model_id=6433
    path="."
    type='Model'
    format='SafeTensor'
    size='full'
    fp='fp16'
    url=f'https://civitai.com/api/download/models/{model_id}?{type}&{format}&{size}&{fp}'
    advanced_dl=advanced_download(model_id=model_id, local_dir=path, type=type, format=format, size=size, fp=fp, token=token)
    assert url==advanced_dl[0]
    assert path==advanced_dl[1]