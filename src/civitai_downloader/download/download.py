from typing import List, Optional, Tuple, Dict
from urllib.parse import urlsplit, parse_qs

from civitai_downloader.api_class import ModelType, ModelFormat, ModelSize, ModelFp, ModelVersionFile
from civitai_downloader.api import CivitAIClient
from civitai_downloader.download import Downloader
from civitai_downloader.download.backend import DownloadManager

base_url='https://civitai.com/api/download/models/'

class FileFilter:
    def __init__(self, type_filter: Optional[ModelType]=None, format_filter: Optional[ModelFormat]=None, size_filter: Optional[ModelSize]=None, fp_filter: Optional[ModelFp]=None):
        self.type_filter=type_filter
        self.format_filter=format_filter
        self.size_filter=size_filter
        self.fp_filter=fp_filter

    @classmethod
    def from_query_params(cls, query_string: str)->'FileFilter':
        params=parse_qs(query_string)
        type_filter=params.get('type', [None])[0]
        format_filter=params.get('format', [None])[0]
        size_filter=params.get('size', [None])[0]
        fp_filter=params.get('fp', [None])[0]

        return cls(type_filter, format_filter, size_filter, fp_filter)
    
    def apply(self, files: List[ModelVersionFile])->List[ModelVersionFile]:
        filtered_files=[]

        for file in files:
            if self._matches_criteria(file):
                filtered_files.append(file)

        return filtered_files
    
    def _matches_criteria(self, file: ModelVersionFile)->bool:
        if self.type_filter and file.type!=self.type_filter: return False
        metadata=file.metadata
        if self.format_filter and metadata.format!=self.format_filter: return False
        if self.size_filter and metadata.size!=self.size_filter: return False
        if self.fp_filter and metadata.fp!=self.fp_filter: return False

        return True
    
class DownloadHandler:
    def __init__(self, api_token: str):
        self.api_token=api_token
        self.api=CivitAIClient(api_token=api_token)
        self.downloader=Downloader(api_token=api_token)

    def process_download(self, files: List[ModelVersionFile], local_dir: str)->Optional[Tuple[str, str, int, str, str]]:
        if not files:
            return None
        
        file=files[0]
        self.downloader.start_download_thread(file, local_dir)
        return (file.downloadUrl, file.name, int(float(file.sizeKB)*1024), local_dir, self.api_token)
    
def _civitai_download(model_version_id: int, local_dir: str, token: str):
    handler=DownloadHandler(token)
    model_version=handler.api.get_model_version(model_version_id)
    if model_version and model_version.files:
        return handler.process_download(model_version.files, local_dir)
    return None

def _advanced_download(model_version_id: int, local_dir: str, token: str, type_filter: ModelType, format_filter: ModelFormat, size_filter: ModelSize, fp_filter: ModelFp):
    handler=DownloadHandler(token)
    model_version=handler.api.get_model_version(model_version_id)
    if model_version:
        file_filter=FileFilter(type_filter, format_filter, size_filter, fp_filter)
        filtered_files=file_filter.apply(model_version.files)
        return handler.process_download(filtered_files, local_dir)
    return None

def _url_download(url: str, local_dir: str, token: str):
    handler=DownloadHandler(token)
    parsed_url=urlsplit(url)

    if parsed_url.scheme!='https' or parsed_url.netloc!='civitai.com': return None

    model_version_id=parsed_url.path.split('/')[-1]
    model_version=handler.api.get_model_version(model_version_id)

    if model_version:
        file_filter=FileFilter.from_query_params(parsed_url.query)
        filtered_files=file_filter.apply(model_version.files)
        return handler.process_download(filtered_files, local_dir)
    
    return None

def _batch_download(model_id: int, local_dir: str, token: str):
    handler=DownloadHandler(api_token=token)
    model=handler.api.get_model(model_id)
    
    if model:
        manager=DownloadManager(model, local_dir, token)
        manager.download_all_files()
        return model, local_dir, token
    return None
    
def _version_batch_download(model_version_id: int, local_dir: str, token: str):
    handler=DownloadHandler(api_token=token)
    model_version=handler.api.get_model_version(model_version_id, token)
    
    if model_version:
        model=model_version.model
        manager=DownloadManager(model, local_dir, token)
        manager.version_download_all_files(model_version_id)
        return model, local_dir, token
    return None