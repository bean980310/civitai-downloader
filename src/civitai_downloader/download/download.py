from civitai_downloader.download import Downloader
from civitai_downloader.download.backend import DownloadManager
from civitai_downloader.api import ModelVersionAPI, ModelAPI
from civitai_downloader.api_class import ModelType, ModelFormat, ModelSize, ModelFp
import time
from urllib.parse import urlsplit

base_url='https://civitai.com/api/download/models/'
    
def civitai_download(model_version_id: int, local_dir: str, token: str):
    api=ModelVersionAPI(api_token=token)
    downloader=Downloader(api_token=token)
    model_version=api.get_model_version_info_from_api(model_version_id)
    if model_version and model_version.files:
        file=model_version.files[0]
        downloader.start_download_thread(file, local_dir)
        return file.downloadUrl, file.name, int(file.sizeKB*1024), local_dir, token
    return None

def advanced_download(model_version_id: int, local_dir: str, token: str, type_filter: ModelType, format_filter: ModelFormat, size_filter: ModelSize, fp_filter: ModelFp):
    api=ModelVersionAPI(api_token=token)
    downloader=Downloader(api_token=token)
    model_version=api.get_model_version_info_from_api(model_version_id)
    if model_version:
        filtered_files=[]
        for file in model_version.files:
            if type_filter and file.type!=type_filter: continue
            metadata=file.metadata
            if format_filter and metadata.format!=format_filter: continue
            if size_filter and metadata.size!=size_filter: continue
            if fp_filter and metadata.fp !=fp_filter: continue

            filtered_files.append(file)
        
        if filtered_files:
            file=filtered_files[0]
            downloader.start_download_thread(file, local_dir)
            return file.downloadUrl, file.name, int(file.sizeKB*1024), local_dir, token
        
        return None

def url_download(url: str, local_dir: str, token: str):
    api=ModelVersionAPI(api_token=token)
    downloader=Downloader(api_token=token)
    splited_url=urlsplit(url)
    if splited_url.scheme!='https' or splited_url.netloc!='civitai.com': return None
    model_version_id=splited_url.path.split('/')[-1]
    if splited_url.query:
        query_params=dict(param.split('=') for param in splited_url.query.split('&'))
        type_filter=query_params.get('type')
        format_filter=query_params.get('format')
        size_filter=query_params.get('size')
        fp_filter=query_params.get('fp')
    model_version=api.get_model_version_info_from_api(model_version_id)

    if model_version:
        filtered_files=[]
        for file in model_version.files:
            if type_filter and file.type!=type_filter: continue
            metadata=file.metadata
            if format_filter and metadata.format!=format_filter: continue
            if size_filter and metadata.size!=size_filter: continue
            if fp_filter and metadata.fp!=fp_filter: continue
            filtered_files.append(file)
            
        if filtered_files:
            file=filtered_files[0]
            downloader.start_download_thread(file, local_dir)
            return file.downloadUrl, file.name, int(file.sizeKB*1024), local_dir, token
        
        return None

def batch_download(model_id: int, local_dir: str, token: str):
    api=ModelAPI(api_token=token)
    model=api.get_model_info_from_api(model_id)
    manager=DownloadManager(model, local_dir, token)
    if model:
        manager.download_all_files()
        return model, local_dir, token
    else:
        return None
    
def version_batch_download(model_version_id: int, local_dir: str, token: str):
    api=ModelVersionAPI(api_token=token)
    model_version=api.get_model_version_info_from_api(model_version_id, token)
    model=model_version.model
    manager=DownloadManager(model, local_dir, token)
    if model_version:
        manager.version_download_all_files(model_version_id)
        return model, local_dir, token