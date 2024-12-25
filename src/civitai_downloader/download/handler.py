from abc import ABC, abstractmethod
import time
import os
from typing import List, Optional, Tuple
import requests
import sys

from civitai_downloader.api.client import CivitAIClient
from civitai_downloader.api_class import ModelVersionFile
from civitai_downloader.download.file_name_extractor import FileNameExtractor
from civitai_downloader.download.util import DownloadUtils
from civitai_downloader.env import JupyterEnvironmentDetector

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class ProgressHandler(ABC):
    @abstractmethod
    def setup(self, filename: str, total_size: int)->None:
        pass

    @abstractmethod
    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        pass

    @abstractmethod
    def finish(self, time_taken: float)->None:
        pass

    @abstractmethod
    def error(self, error_message: str)->None:
        pass

class NotebookProgressHandler(ProgressHandler):
    """주피터 노트북 / 코랩에서 ipywidgets 기반 진행률 표시."""
    def __init__(self):
        self.widgets, self.display = JupyterEnvironmentDetector.get_ipywidgets()
        self.is_colab = JupyterEnvironmentDetector.in_colab()
        self.progress_bar = None
        self.status_label = None
        self.file_label = None

    def setup(self, filename: str, total_size: int)->None:
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

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        self.progress_bar.value = downloaded
        progress_percentage = (downloaded / total_size * 100) if total_size > 0 else 0
        speed = (downloaded / elapsed_time) if elapsed_time>0 else 0
        speed_str = f'{speed/(1024**2):.2f} MB/s'
        downloaded_str = DownloadUtils.format_bytes(downloaded)
        total_size_str = DownloadUtils.format_bytes(total_size)
        remaining = (total_size - downloaded) / speed if speed>0 else 0
        self.status_label.value = (
            f"<b>{progress_percentage:.2f}%</b> "
            f"({downloaded_str}/{total_size_str}) "
            f"[{speed_str}, Elapsed: {DownloadUtils.format_time(elapsed_time)}"
            f"<{DownloadUtils.format_time(remaining)}]"
        )

    def finish(self, time_taken: float)->None:
        self.progress_bar.bar_style = 'success'
        self.status_label.value = f'<b>Download completed!</b> (Total Time: {DownloadUtils.format_time(time_taken)})'

    def error(self, error_message: str)->None:
        self.progress_bar.bar_style = 'danger'
        self.status_label.value = f'<b>Error:</b> {error_message}'

class TqdmProgressHandler(ProgressHandler):
    """tqdm 기반 일반 터미널(or ipynb) 진행률 표시."""
    def __init__(self):
        self.progress_bar = None

    def setup(self, filename: str, total_size: int)->None:
        print(f"\nDownloading: {filename}")
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        self.progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            bar_format=bar_format,
        )

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        if self.progress_bar is not None:
            self.progress_bar.update(chunk_size)
            speed = (downloaded / elapsed_time) if elapsed_time>0 else 0
            self.progress_bar.set_postfix({
                'speed': f"{speed/(1024**2):.2f} MB/s"
            })

    def finish(self, time_taken: float)->None:
        if self.progress_bar is not None:
            self.progress_bar.close()
        print(f"Download finished in {DownloadUtils.format_time(time_taken)}")

    def error(self, error_message: str)->None:
        if self.progress_bar is not None:
            self.progress_bar.close()
        print(f"Error: {error_message}")

class ConsoleProgressHandler(ProgressHandler):
    """tqdm 미설치 시, 간단한 콘솔 진행률 표시."""
    def __init__(self):
        self.filename = None
        self.total_size_str = None

    def setup(self, filename: str, total_size: int)->None:
        self.filename = filename
        self.total_size_str = DownloadUtils.format_bytes(total_size)
        print(f"\nDownloading: {filename}")

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        speed = (downloaded / elapsed_time) if elapsed_time>0 else 0
        speed_str = f"{speed/(1024**2):.2f} MB/s"
        if total_size>0:
            progress_percentage = (downloaded / total_size)*100
            sys.stdout.write(f"\r{self.filename} [{progress_percentage:.2f}%] - {speed_str}")
        else:
            sys.stdout.write(f"\r{self.filename} [Downloaded={DownloadUtils.format_bytes(downloaded)}, {speed_str}]")
        sys.stdout.flush()

    def finish(self, time_taken: float)->None:
        sys.stdout.write("\n")
        print(f"Download finished in {DownloadUtils.format_time(time_taken)}")

    def error(self, error_message: str)->None:
        sys.stdout.write("\n")
        print(f"Error: {error_message}")

class DownloadHandler:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.api = CivitAIClient(api_token=api_token)

    def process_download(self, files: List[ModelVersionFile], local_dir: str) -> Optional[Tuple[str, str, int, str, str]]:
        """
        파일 목록 중 첫 번째 파일만 받아온 뒤,
        (downloadUrl, 파일명, 파일 크기, 로컬 경로, 토큰) 튜플을 반환 (기존 코드와 호환).
        """
        if not files:
            return None

        target_file = files[0]
        # 실제 다운로드 실행
        self._download_file(target_file, local_dir)

        return (
            target_file.downloadUrl,
            target_file.name,
            int(float(target_file.sizeKB)*1024),
            local_dir,
            self.api_token
        )

    def _download_file(self, file: ModelVersionFile, local_dir: str):
        """
        실제 다운로드 로직을 여기서 처리.
        - URL 먼저 파일명 추출
        - 진행률 표시를 위한 ProgressHandler 사용
        """
        url = file.downloadUrl
        if not url.startswith('https://'):
            print(f"Invalid URL: {url}")
            return

        # 1) URL로부터 파일명 추출 (우선순위)
        if not file.name:
            extracted = FileNameExtractor.from_url(url)
            if extracted:
                file.name = extracted
                print(f"Filename found via URL parsing -> {file.name}")

        # 2) 그래도 file.name이 없다면, 필요한 경우 API 등으로 조회하거나 fallback
        if not file.name:
            file.name = "untitled.bin"

        save_path = os.path.join(local_dir, file.name)
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(save_path):
            print(f"[Skip] Already exists: {file.name}")
            return

        # ProgressHandler 인스턴스 생성
        progress_handler = self._get_progress_handler()

        # 다운로드 시작
        headers = {"Authorization": f"Bearer {self.api_token}"}
        start_time = time.time()
        downloaded = 0

        try:
            resp = requests.get(url, headers=headers, stream=True)
            resp.raise_for_status()
            total_size = int(resp.headers.get('content-length', 0))

            progress_handler.setup(file.name, total_size)

            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    progress_handler.update(len(chunk), downloaded, total_size, elapsed)
            
            resp.close()
            progress_handler.finish(time.time() - start_time)
            print(f"Downloaded: {save_path}")

        except Exception as e:
            progress_handler.error(str(e))

    def _get_progress_handler(self):
        """주피터/코랩/터미널 환경에 따라 적절한 진행률 표시 핸들러를 반환"""
        from civitai_downloader.env import JupyterEnvironmentDetector
        widgets, _ = JupyterEnvironmentDetector.get_ipywidgets()

        is_notebook = JupyterEnvironmentDetector.in_jupyter_notebook()
        is_colab = JupyterEnvironmentDetector.in_colab()

        if widgets and (is_notebook or is_colab):
            return NotebookProgressHandler()
        elif tqdm:
            return TqdmProgressHandler()
        else:
            return ConsoleProgressHandler()