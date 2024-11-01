import threading
from typing import Optional
from urllib.parse import urljoin
from civitai_downloader.download.backend import download_file
from civitai_downloader.api.model import get_model_info_from_api, get_model_version_info_from_api

base_url='https://civitai.com/api/download/models/'
    
def civitai_download(model_id: int, local_dir: str, token: str):
    url = urljoin(base_url, str(model_id))
    start_download_thread(url, local_dir, token)
    return url, local_dir, token

def advanced_download(model_id: int, version_id: int, local_dir: str, token: str, format: enumerate, size: enumerate, fp: enumerate, type: Optional[enumerate]='Model'):
    url=f'{base_url}{model_id}/{version_id}?type={type}&format={format}&size={size}&fp={fp}'
    start_download_thread(url, local_dir, token)
    return url, local_dir, token

def url_download(url: str, local_dir: str, token: str):
    start_download_thread(url, local_dir, token)
    return url, local_dir, token

def batch_download(model_id: int, local_dir: str, token: str):
    model_info=get_model_info_from_api(model_id, token)
    if model_info:
        get_download_files(model_info, local_dir, token)
        return model_info, local_dir, token
    else:
        return None
    
def version_batch_download(model_id: int, local_dir: str, token: str):
    model_version_info=get_model_version_info_from_api(model_id, token)
    if model_version_info:
        get_download_files(model_version_info, local_dir, token)
        return model_version_info, local_dir, token

def start_download_thread(url: str, local_dir: str, token: str):
    thread = threading.Thread(target=download_file, args=(url, local_dir, token))
    thread.start()

def get_download_files(model_info, local_dir, token):
    download_files=[]
    for version in model_info.get('modelVersions', []):
        for file in version.get('files', []):
            download_url=file.get('downloadUrl')
            if download_url:
                start_download_thread(download_url, local_dir, token)
                download_files.append(download_url)

def get_version_download_files(model_version_info, local_dir, token):
    download_files=[]
    for file in model_version_info.get('files', []):
        download_url=file.get('downloadUrl')
        if download_url:
            start_download_thread(download_url, local_dir, token)
            download_files.append(download_url)