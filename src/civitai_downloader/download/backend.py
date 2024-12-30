import os.path
import sys
import time
import requests
import threading
import tempfile
import shutil
from abc import ABC, abstractmethod
from typing import Optional

from civitai_downloader.api_class import ModelVersionFile
from civitai_downloader.api import CivitAIClient
from civitai_downloader.env import JupyterEnvironmentDetector
from civitai_downloader.download.util import DownloadUtils
from civitai_downloader.download.file_name_extractor import FileNameExtractor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ----------------------------------------------------------------------------
# (A) ProgressHandler + 노트북/콘솔용 구현체들
#    (기존 DownloadHandler 내부에서 하던 '진행률 표시' 기능)
# ----------------------------------------------------------------------------
class ProgressHandler(ABC):
    @abstractmethod
    def setup(self, filename: str, total_size: int) -> None:
        pass

    @abstractmethod
    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float) -> None:
        pass

    @abstractmethod
    def finish(self, time_taken: float) -> None:
        pass

    @abstractmethod
    def error(self, error_message: str) -> None:
        pass


class NotebookProgressHandler(ProgressHandler):
    """
    주피터 노트북 / 구글 코랩 환경에서 ipywidgets 기반 진행률 표시
    """
    def __init__(self):
        self.widgets, self.display = JupyterEnvironmentDetector.get_ipywidgets()
        self.is_colab = JupyterEnvironmentDetector.in_colab()
        self.progress_bar = None
        self.status_label = None
        self.file_label = None

    def setup(self, filename: str, total_size: int) -> None:
        self.file_label = self.widgets.HTML(value=f'<b>Downloading</b> {filename}')
        self.progress_bar = self.widgets.IntProgress(
            value=0,
            min=0,
            max=total_size if total_size>0 else 1,
            bar_style='info',
            orientation='horizontal',
            layout=self.widgets.Layout(width='100%' if self.is_colab else '100')
        )
        self.status_label = self.widgets.HTML(value="0%")
        progress_info = self.widgets.HBox([self.progress_bar, self.status_label])
        progress_box = self.widgets.VBox([self.file_label, progress_info])
        self.display(progress_box)

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float) -> None:
        self.progress_bar.value = downloaded
        progress_percentage = (downloaded / total_size)*100 if total_size > 0 else 0
        speed = downloaded / elapsed_time if elapsed_time > 0 else 0

        speed_str = f'{speed/(1024**2):.2f} MB/s'
        downloaded_str = DownloadUtils.format_bytes(downloaded)
        total_size_str = DownloadUtils.format_bytes(total_size)
        elapsed_time_str = DownloadUtils.format_time(elapsed_time)
        remaining_time = (total_size - downloaded)/speed if speed > 0 else 0
        remaining_time_str = DownloadUtils.format_time(remaining_time)

        self.status_label.value = (
            f"<b>{progress_percentage:.2f}%</b> ({downloaded_str}/{total_size_str}) "
            f"[{speed_str}, {elapsed_time_str}<{remaining_time_str}]"
        )

    def finish(self, time_taken: float) -> None:
        self.progress_bar.bar_style = 'success'
        self.status_label.value = f'<b>Downloaded</b> (Total Time: {DownloadUtils.format_time(time_taken)})'

    def error(self, error_message: str) -> None:
        if self.progress_bar:
            self.progress_bar.bar_style = 'danger'
        if self.status_label:
            self.status_label.value = f'<b>Error</b> {error_message}'


class TqdmProgressHandler(ProgressHandler):
    """
    tqdm 기반 진행률 표시 (터미널/IPython 양쪽에서 사용 가능)
    """
    def __init__(self):
        self.progress_bar = None

    def setup(self, filename: str, total_size: int) -> None:
        print(f"downloading: {filename}")
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        self.progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            ncols=None,
            bar_format=bar_format
        )

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float) -> None:
        if self.progress_bar:
            self.progress_bar.update(chunk_size)
            speed = downloaded / elapsed_time if elapsed_time > 0 else 0
            speed_str = f"{speed/(1024**2):.2f} MB/s"

            if total_size > 0:
                progress_percentage = (downloaded / total_size)*100
                self.progress_bar.set_postfix({
                    'percent': f'{progress_percentage:.2f}%',
                    'speed': speed_str
                })
            else:
                self.progress_bar.set_postfix({
                    'downloaded': DownloadUtils.format_bytes(downloaded),
                    'speed': speed_str
                })

    def finish(self, time_taken: float) -> None:
        if self.progress_bar:
            self.progress_bar.close()
        print(f"Finished in {DownloadUtils.format_time(time_taken)}")

    def error(self, error_message: str) -> None:
        if self.progress_bar:
            self.progress_bar.close()
        print(f"\nError: {error_message}")


class ConsoleProgressHandler(ProgressHandler):
    """
    일반 콘솔(터미널) 환경에서 간단한 진행률 표시
    (tqdm 미설치 시 사용)
    """
    def __init__(self):
        self.filename = None
        self.total_size_str = None

    def setup(self, filename: str, total_size: int) -> None:
        self.filename = filename
        self.total_size_str = DownloadUtils.format_bytes(total_size)
        print(f"\nDownloading: {filename}")

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float) -> None:
        speed = downloaded / elapsed_time if elapsed_time > 0 else 0
        speed_str = f"{speed/(1024**2):.2f} MB/s"
        if total_size > 0:
            progress_percentage = (downloaded / total_size)*100
            downloaded_str = DownloadUtils.format_bytes(downloaded)
            sys.stdout.write(
                f"\r{self.filename} - {progress_percentage:.2f}% "
                f"({downloaded_str}/{self.total_size_str}, {speed_str})"
            )
        else:
            downloaded_str = DownloadUtils.format_bytes(downloaded)
            sys.stdout.write(
                f"\r{self.filename} - Downloaded: {downloaded_str}, Speed: {speed_str}"
            )
        sys.stdout.flush()

    def finish(self, time_taken: float) -> None:
        sys.stdout.write("\n")
        print(f"Download completed: {self.filename}")
        print(f"Time taken: {DownloadUtils.format_time(time_taken)}")

    def error(self, error_message: str) -> None:
        sys.stdout.write('\n')
        print(f"\nError: {error_message}")


# ----------------------------------------------------------------------------
# (B) Downloader 클래스 : (단일/멀티) 파일 다운로드 로직
#     -> DownloadHandler 없이도 여기서 직접 ProgressHandler 사용
# ----------------------------------------------------------------------------
class Downloader:
    CHUNK_SIZE = 1638400
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

    def __init__(self, api_token: str, use_cache: bool=True, cache_dir: Optional[str]=None):
        """
        api_token : civitai API 토큰
        cache_dir : 사용자 지정 임시(캐시) 디렉토리 경로.
                    None이면 tempfile.mkdtemp() 등으로 다운로드 시마다 임시 디렉토리를 생성/삭제
        """
        self.api_token = api_token
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_progress_handler(self) -> ProgressHandler:
        # Jupyter/Colab인 경우 NotebookProgressHandler,
        # tqdm 설치되어 있으면 TqdmProgressHandler,
        # 아니면 ConsoleProgressHandler
        widgets, _ = JupyterEnvironmentDetector.get_ipywidgets()
        is_notebook = JupyterEnvironmentDetector.in_jupyter_notebook()
        is_colab = JupyterEnvironmentDetector.in_colab()

        if widgets and (is_notebook or is_colab):
            return NotebookProgressHandler()
        elif tqdm:
            return TqdmProgressHandler()
        else:
            return ConsoleProgressHandler()

    def start_download_thread(self, file: ModelVersionFile, local_dir: str, overwrite: bool=False) -> threading.Thread:
        """
        여러 파일 다운로드를 병렬(멀티스레드)로 처리하기 위해
        Thread를 시작하는 메서드 예시.
        """
        t = threading.Thread(
            target=self._download_file,
            args=(file, local_dir, overwrite)
        )
        t.start()
        return t

    def _download_file(self, file: ModelVersionFile, save_dir: str, overwrite: bool=False) -> None:
        """
        실제 다운로드 진행 (ProgressHandler로 진행률 표시).
        """
        url = file.downloadUrl
        if not url.startswith('https://'):
            print(f"Invalid URL: {url}")
            return

        # (1) URL에서 파일명 우선 추출
        if not file.name:
            extracted_filename = FileNameExtractor.from_url(url)
            if extracted_filename:
                file.name = extracted_filename
                print(f"Filename found via URL parsing: {file.name}")

        # (2) 그래도 file.name이 없으면 fallback
        if not file.name:
            file.name = "untitled.bin"

        output_file = os.path.join(save_dir, file.name)
        os.makedirs(save_dir, exist_ok=True)

        # 덮어쓰기 옵션이 False이고, 파일이 이미 존재하면 스킵
        if os.path.exists(output_file) and not overwrite:
            print(f"File already exists: {file.name}")
            return

        # (3) requests로 다운로드
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'User-Agent': self.USER_AGENT,
        }
        progress_handler = self._get_progress_handler()
        downloaded = 0
        start_time = time.time()

        try:
            r = requests.get(url, headers=headers, stream=True)
            r.raise_for_status()
        except Exception as e:
            progress_handler.error(str(e))
            return

        total_size = int(r.headers.get('content-length', 0))
        progress_handler.setup(file.name, total_size)
        print(f"Downloading: {url}")

        if self.use_cache:
            if not self.cache_dir:
                self.cache_dir = os.path.join(save_dir, ".civitai", "download")
                os.makedirs(self.cache_dir, exist_ok=True)
            temp_filepath = os.path.join(self.cache_dir, f"{file.name}.download")
            temp_dir=None
        else:
            # 매번 임시 디렉토리 생성 -> 다운로드 끝나면 삭제
            temp_dir = tempfile.mkdtemp(prefix='civitai_cache_')
            temp_filepath = os.path.join(temp_dir, file.name)

        try:
            with open(temp_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=self.CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    progress_handler.update(len(chunk), downloaded, total_size, elapsed)
            r.close()
            shutil.move(temp_filepath, output_file)
        finally:
            if not self.cache_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        time_taken = time.time() - start_time
        progress_handler.finish(time_taken)
        print(f"Saved as: {output_file}\n")