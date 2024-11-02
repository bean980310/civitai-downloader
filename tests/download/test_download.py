from civitai_downloader.download.download import civitai_download, advanced_download
from civitai_downloader.token.token import get_token, prompt_for_civitai_token
from civitai_downloader.api.model import get_model_version_info_from_api

token=get_token()
if token is None:
    token=prompt_for_civitai_token()

def test_civitai_download():
    model_id=215522
    path="."
    model_version_info=get_model_version_info_from_api(model_id, token)
    for files in model_version_info.get('files', []):
        url=files.get('downloadUrl')
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
    model_version_info=get_model_version_info_from_api(model_id, token)
    filtered_files=[]
    for file in model_version_info.get('files', []):
        if file.get('type')!=type: continue
        metadata=file.get('metadata')
        if metadata.get('format')!=format: continue
        if metadata.get('size')!=size: continue
        if metadata.get('fp')!=fp: continue
        filtered_files.append(file)
    for file in filtered_files:
        assert file.get('type')==type
        metadata=file.get('metadata')
        assert metadata.get('format')==format
        assert metadata.get('size')==size
        assert metadata.get('fp')==fp
        url=file.get('downloadUrl')
    advanced_dl=advanced_download(model_id=model_id, local_dir=path, type=type, format=format, size=size, fp=fp, token=token)
    assert url==advanced_dl[0]
    assert path==advanced_dl[1]