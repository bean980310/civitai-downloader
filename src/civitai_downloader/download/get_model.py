import requests
from civitai_downloader.api.model import get_model_version_info_from_api
from civitai_downloader.download.download import get_version_download_file

def get_model_version_info_for_download(model_id, local_dir, token):
    model_version_info=get_model_version_info_from_api(model_id, token)