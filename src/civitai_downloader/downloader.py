from civitai_downloader.download import _civitai_download, _advanced_download, _url_download, _batch_download, _version_batch_download
from civitai_downloader.api_class import ModelType, ModelFormat, ModelSize, ModelFp
from civitai_downloader import login

def civitai_download(model_version_id: int, local_dir: str, token: str=login(), use_cache: bool=True, cache_dir: str=None):
    _civitai_download(model_version_id=model_version_id, local_dir=local_dir, token=token, use_cache=use_cache, cache_dir=cache_dir)

def advanced_download(model_version_id: int, local_dir: str, type_filter: ModelType, format_filter: ModelFormat, size_filter: ModelSize, fp_filter: ModelFp, token: str=login(), use_cache: bool=True, cache_dir: str=None):
    _advanced_download(model_version_id=model_version_id, local_dir=local_dir, token=token, type_filter=type_filter, format_filter=format_filter, size_filter=size_filter, fp_filter=fp_filter, use_cache=use_cache, cache_dir=cache_dir)

def url_download(url: str, local_dir: str, token: str=login(),use_cache: bool=True, cache_dir: str=None):
    _url_download(url=url, local_dir=local_dir, token=token, use_cache=use_cache, cache_dir=cache_dir)

def batch_download(model_id: int, local_dir: str, token: str=login(),use_cache: bool=True, cache_dir: str=None):
    _batch_download(model_id=model_id, local_dir=local_dir, token=token, use_cache=use_cache, cache_dir=cache_dir)

def version_batch_download(model_version_id: int, local_dir: str, token: str=login(),use_cache: bool=True, cache_dir: str=None):
    _version_batch_download(model_version_id=model_version_id, local_dir=local_dir, token=token, use_cache=use_cache, cache_dir=cache_dir)
