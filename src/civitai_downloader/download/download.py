# download.py
import os
from typing import Optional, Tuple
from urllib.parse import urlsplit, parse_qs

from civitai_downloader.api_class import ModelType, ModelFormat, ModelSize, ModelFp, ModelVersionFile
from civitai_downloader.api.client import CivitAIClient
from civitai_downloader.download.file_name_extractor import FileNameExtractor
from civitai_downloader.download.backend import Downloader  # or from .manager import DownloadManager
# ↑ 프로젝트 구조에 따라 임포트 경로 조정
# 만약 Manager 기반으로 처리하고 싶다면: from civitai_downloader.download.manager import DownloadManager

base_url = 'https://civitai.com/api/download/models/'

# -----------------------------------
# FileFilter: _advanced_download에서 쓰이는 필터
# -----------------------------------
class FileFilter:
    def __init__(
        self,
        type_filter: Optional[ModelType] = None,
        format_filter: Optional[ModelFormat] = None,
        size_filter: Optional[ModelSize] = None,
        fp_filter: Optional[ModelFp] = None
    ):
        self.type_filter = type_filter
        self.format_filter = format_filter
        self.size_filter = size_filter
        self.fp_filter = fp_filter

    @classmethod
    def from_query_params(cls, query_string: str) -> 'FileFilter':
        params = parse_qs(query_string)
        type_filter   = params.get('type',   [None])[0]
        format_filter = params.get('format', [None])[0]
        size_filter   = params.get('size',   [None])[0]
        fp_filter     = params.get('fp',     [None])[0]
        return cls(type_filter, format_filter, size_filter, fp_filter)

    def apply(self, files: list[ModelVersionFile]) -> list[ModelVersionFile]:
        return [f for f in files if self._matches_criteria(f)]

    def _matches_criteria(self, file: ModelVersionFile) -> bool:
        if self.type_filter and file.type != self.type_filter:
            return False
        if not file.metadata:
            return False
        if self.format_filter and file.metadata.format != self.format_filter:
            return False
        if self.size_filter and file.metadata.size != self.size_filter:
            return False
        if self.fp_filter and file.metadata.fp != self.fp_filter:
            return False
        return True

# --------------------------------------------
# (1) _civitai_download : 기존 단일 다운로드
# --------------------------------------------
def _civitai_download(model_version_id: int, local_dir: str, token: str, use_cache: bool=True, cache_dir: Optional[str]=None):
    client = CivitAIClient(api_token=token)
    url = f"{base_url}{model_version_id}"
    extracted_filename = FileNameExtractor.from_url(url)

    # Downloader (단일 파일 다운로드) 사용
    downloader = Downloader(api_token=token, use_cache=use_cache, cache_dir=cache_dir)

    if extracted_filename:
        # URL로부터 파일명 추출 성공
        fake_file = ModelVersionFile(
            downloadUrl=url,
            name=extracted_filename,
            sizeKB=0.0,
            type=None,
            metadata=None
        )
        # downloader로 단일 파일 다운로드
        downloader._download_file(fake_file, local_dir)
        return f"{local_dir}/{extracted_filename}"

    # 실패 -> API 요청하여 model_version.files를 가져옴
    model_version = client.get_model_version(model_version_id)
    if model_version and model_version.files:
        target_file = model_version.files[0]
        downloader._download_file(target_file, local_dir)
        return f"{local_dir}/{extracted_filename}"
    
    print("No file found to download.")
    return None

# ---------------------------------------------
# (2) _advanced_download : 유지하고 싶은 함수
# ---------------------------------------------
def _advanced_download(
    model_version_id: int,
    local_dir: str,
    token: str,
    type_filter: ModelType,
    format_filter: ModelFormat,
    size_filter: ModelSize,
    fp_filter: ModelFp,
    use_cache: bool=True,
    cache_dir: Optional[str]=None
):
    """
    model_version_id로부터 모델 버전을 조회한 다음,
    주어진 필터(type_filter, format_filter 등)를 적용해 일치하는 파일들만 다운로드.
    """
    client = CivitAIClient(api_token=token)
    url = f"{base_url}{model_version_id}"
    extracted_filename = FileNameExtractor.from_url(url)

    downloader = Downloader(api_token=token, use_cache=use_cache, cache_dir=cache_dir)

    if extracted_filename:
        # URL로부터 파일명 추출 -> 곧바로 다운로드 시도
        fake_file = ModelVersionFile(
            downloadUrl=url,
            name=extracted_filename,
            sizeKB=0.0,
            type=None,
            metadata=None
        )
        downloader._download_file(fake_file, local_dir)
        return f"{local_dir}/{extracted_filename}"

    # 추출 실패 -> model_version 조회
    model_version = client.get_model_version(model_version_id)
    if not model_version:
        return None

    # (a) 파일 필터 적용
    file_filter = FileFilter(type_filter, format_filter, size_filter, fp_filter)
    filtered_files = file_filter.apply(model_version.files)
    if not filtered_files:
        print("No matching files found with the given filter.")
        return None

    # (b) 여기서는 일단 "첫 번째 파일"만 다운로드 (기존과 동일)
    target_file = filtered_files[0]
    downloader._download_file(target_file, local_dir)
    return f"{local_dir}/{extracted_filename}"

# -----------------------------------------------
# (3) _url_download, _batch_download, etc. 예시
# -----------------------------------------------
def _url_download(url: str, local_dir: str, token: str, use_cache: bool=True, cache_dir: Optional[str]=None):
    client = CivitAIClient(api_token=token)
    parsed_url = urlsplit(url)
    if parsed_url.scheme != 'https' or parsed_url.netloc != 'civitai.com':
        return None

    downloader = Downloader(api_token=token, use_cache=use_cache, cache_dir=cache_dir)
    extracted_filename = FileNameExtractor.from_url(url)
    if extracted_filename:
        # 그대로 단일 다운로드
        fake_file = ModelVersionFile(
            downloadUrl=url,
            name=extracted_filename,
            sizeKB=0.0,
            type=None,
            metadata=None
        )
        downloader._download_file(fake_file, local_dir)
        return f"{local_dir}/{extracted_filename}"

    # or else: ...
    model_version_id = parsed_url.path.split('/')[-1]
    model_version = client.get_model_version(model_version_id)
    if not model_version:
        return None

    # query-based filter
    file_filter = FileFilter.from_query_params(parsed_url.query)
    filtered_files = file_filter.apply(model_version.files)
    if filtered_files:
        target_file = filtered_files[0]
        downloader._download_file(target_file, local_dir)
        return f"{local_dir}/{extracted_filename}"
    
    return None

def _batch_download(model_id: int, local_dir: str, token: str, use_cache: bool=True, cache_dir: Optional[str]=None):
    """
    모델 ID로 전체 버전을 순회하며 모든 파일 다운로드 (예시).
    만약 스레드 병렬 처리를 원하면 `DownloadManager`를 쓸 수도 있음.
    """
    client = CivitAIClient(api_token=token)
    model = client.get_model(model_id)
    if not model:
        return None

    # 여기를 DownloadManager로 바꿔도 OK
    # from civitai_downloader.download.manager import DownloadManager
    # manager = DownloadManager(model, local_dir, token)
    # manager.download_all_files()
    # return model, local_dir, token

    # 여기서는 간단히 '각 버전의 각 파일'을 하나씩 Downloader._download_file()로 다운로드하는 예시
    downloader = Downloader(api_token=token, use_cache=use_cache, cache_dir=cache_dir)
    for version in model.modelVersions:
        for file_data in version.files:
            file_obj = ModelVersionFile(**file_data)
            downloader._download_file(file_obj, local_dir)

    print("All file downloaded!")
    return

def _version_batch_download(model_version_id: int, local_dir: str, token: str, use_cache: bool=True, cache_dir: Optional[str]=None):
    """
    특정 버전 ID에 해당하는 모든 파일 다운로드.
    """
    client = CivitAIClient(api_token=token)
    model_version = client.get_model_version(model_version_id)
    if not model_version:
        return None

    downloader = Downloader(api_token=token, use_cache=use_cache, cache_dir=cache_dir)
    for file_data in model_version.files:
        file_obj = ModelVersionFile(**file_data)
        downloader._download_file(file_obj, local_dir)
    print("All file downloaded!")
    return