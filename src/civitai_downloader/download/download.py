import threading
from urllib.parse import urlsplit
from civitai_downloader.download.backend import download_file
from civitai_downloader.api.model import get_model_info_from_api, get_model_version_info_from_api
import time

base_url='https://civitai.com/api/download/models/'
    
def civitai_download(model_id: int, local_dir: str, token: str):
    model_version_info=get_model_version_info_from_api(model_id, token)
    if model_version_info:
        url=model_version_info.get('downloadUrl')
        filename=model_version_info.get('files')[0].get('name')
        filesize_kb=model_version_info.get('files')[0].get('sizeKB', 0)
        filesize=int(float(filesize_kb)*1024)
        start_download_thread(url, filename, filesize, local_dir, token)
        return url, filename, filesize, local_dir, token

def advanced_download(model_id: int, local_dir: str, token: str, type: enumerate, format: enumerate, size: enumerate, fp: enumerate):
    model_version_info=get_model_version_info_from_api(model_id, token)
    if model_version_info:
        filtered_files=[]
        for file in model_version_info.get('files', []):
            if type and file.get('type')!=type: continue
            metadata=file.get('metadata')
            if format and metadata.get('format')!=format: continue
            if size and metadata.get('size')!=size: continue
            if fp and metadata.get('fp')!=fp: continue
            filtered_files.append(file)
        for file in filtered_files:
            url=file.get('downloadUrl')
            filename=file.get('name')
            filesize_kb=file.get('sizeKB', 0)
            filesize=int(float(filesize_kb)*1024)
            start_download_thread(url, filename, filesize, local_dir, token)
            return url, filename, filesize, local_dir, token

def url_download(url: str, local_dir: str, token: str):
    splited_url=urlsplit(url)
    if splited_url.scheme!='https' or splited_url.netloc!='civitai.com': return None
    model_id=splited_url.path.split('/')[-1]
    if splited_url.query:
        type=splited_url.query.split('&')[0].split('=')[1]
        format=splited_url.query.split('&')[1].split('=')[1]
        size=splited_url.query.split('&')[2].split('=')[1]
        fp=splited_url.query.split('&')[3].split('=')[1]
    model_version_info=get_model_version_info_from_api(model_id, token)
    if model_version_info:
        filtered_files=[]
        for file in model_version_info.get('files', []):
            if type and file.get('type')!=type: continue
            metadata=file.get('metadata')
            if format and metadata.get('format')!=format: continue
            if size and metadata.get('size')!=size: continue
            if fp and metadata.get('fp')!=fp: continue
            filtered_files.append(file)
        for file in filtered_files:
            url=file.get('downloadUrl')
            filename=file.get('name')
            filesize_kb=file.get('sizeKB', 0)
            filesize=int(float(filesize_kb)*1024)
            start_download_thread(url, filename, filesize, local_dir, token)
            return url, filename, filesize, local_dir, token

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

def start_download_thread(url: str, filename: str, filesize: int, local_dir: str, token: str):
    thread = threading.Thread(target=download_file, args=(url, filename, filesize, local_dir, token))
    thread.start()

def get_download_files(model_info, local_dir, token):
    download_files=[]
    for version in model_info.get('modelVersions', []):
        for file in version.get('files', []):
            download_url=file.get('downloadUrl')
            filename=file.get('name')
            filesize_kb=file.get('sizeKB', 0)
            filesize=int(float(filesize_kb)*1024)
            if download_url:
                start_download_thread(download_url, filename, filesize, local_dir, token)
                download_files.append(download_url)
                return download_url, filename, filesize, local_dir, token
            time.sleep(1)
        time.sleep(1)

def get_version_download_files(model_version_info, local_dir, token):
    download_files=[]
    for file in model_version_info.get('files', []):
        download_url=file.get('downloadUrl')
        filename=file.get('name')
        filesize_kb=file.get('sizeKB', 0)
        filesize=int(float(filesize_kb)*1024)
        if download_url:
            start_download_thread(download_url, filename, filesize, local_dir, token)
            download_files.append(download_url)
            return download_url, filename, filesize, local_dir, token
        time.sleep(1)