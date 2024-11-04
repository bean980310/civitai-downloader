import os
from civitai_downloader.download.download import civitai_download, advanced_download
from civitai_downloader.token.token import get_token
from civitai_downloader.api.model_version import ModelVersionAPI

token=get_token()

def test_civitai_download():
    model_version_id=215522
    path="."
    api=ModelVersionAPI(api_token=token)
    model=api.get_model_version_info_from_api(model_version_id)
    url=model.downloadUrl
    if model:
        file=model.files[0]
    civitai_download(model_version_id=model_version_id, local_dir=path, token=token)
    file_exists=os.path.exists(os.path.join(path, file.name))
    assert model.id==model_version_id
    assert model.downloadUrl==url
    assert file_exists

def test_advanced_download():
    model_version_id=6433
    path="."
    type='Model'
    format='SafeTensor'
    size='full'
    fp='fp16'
    api=ModelVersionAPI(api_token=token)
    model=api.get_model_version_info_from_api(model_version_id)
    filtered_files=[]
    for file in model.files:
        if file.type!=type: continue
        metadata=file.metadata
        if metadata.format!=format: continue
        if metadata.size!=size: continue
        if metadata.fp!=fp: continue
        filtered_files.append(file)
    for file in filtered_files:
        assert file.type==type
        metadata=file.metadata
        assert metadata.format==format
        assert metadata.size==size
        assert metadata.fp==fp
    advanced_download(model_version_id=model_version_id, local_dir=path, type_filter=type, format_filter=format, size_filter=size, fp_filter=fp, token=token)
    file_exists=os.path.exists(os.path.join(path, file.name))
    assert file_exists