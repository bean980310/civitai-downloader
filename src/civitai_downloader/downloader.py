from civitai_downloader.download import _civitai_download, _advanced_download, _url_download, _batch_download, _version_batch_download
from civitai_downloader.api_class import ModelType, ModelFormat, ModelSize, ModelFp
from civitai_downloader import login

def civitai_download(model_version_id: int, local_dir: str):
    token=login()
    return _civitai_download(model_version_id, local_dir, token)

def advanced_download(model_version_id: int, local_dir: str, type_filter: ModelType, format_filter: ModelFormat, size_filter: ModelSize, fp_filter: ModelFp):
    token=login()
    return _advanced_download(model_version_id, local_dir, token, type_filter, format_filter, size_filter, fp_filter)

def url_download(url: str, local_dir: str):
    token=login()
    return _url_download(url, local_dir, token)

def batch_download(model_id: int, local_dir: str):
    token=login()
    return _batch_download(model_id, local_dir, token)

def version_batch_download(model_version_id: int, local_dir: str):
    token=login()
    return _version_batch_download(model_version_id, local_dir, token)
