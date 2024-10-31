import threading
import requests
from urllib.parse import urljoin
from civitai_downloader.download.backend import download_file

CIVITAI_URL='https://civitai.com/'
CIVITAI_DOWNLOAD_API_URL=urljoin(CIVITAI_URL, 'api/download/models/')
CIVITAI_MODELS_URL=urljoin(CIVITAI_URL, 'models/')

civitai_url=CIVITAI_URL
base_url=CIVITAI_DOWNLOAD_API_URL
civitai_models_url=CIVITAI_MODELS_URL
    
def civitai_download(model_id: int, local_dir: str, token: str):
    url = urljoin(base_url, str(model_id))
    start_download_thread(url, local_dir, token)
    return url, local_dir, token

def url_download(url: str, local_dir: str, token: str):
    start_download_thread(url, local_dir, token)
    return url, local_dir, token

def start_download_thread(url: str, local_dir: str, token: str):
    thread = threading.Thread(target=download_file, args=(url, local_dir, token))
    thread.start()
