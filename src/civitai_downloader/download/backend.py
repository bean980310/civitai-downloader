import os.path
import sys
import time
import requests
import threading
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO

from civitai_downloader.api_class import ModelVersionFile
from civitai_downloader.api import CivitAIClient
from civitai_downloader.env import JupyterEnvironmentDetector
from civitai_downloader.download.util import DownloadUtils

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
    def __init__(self):
        self.widgets, self.display=JupyterEnvironmentDetector.get_ipywidgets()
        self.is_colab=JupyterEnvironmentDetector.in_colab()
        self.progress_bar=None
        self.status_label=None
        self.file_label=None

    def setup(self, filename: str, total_size: int)->None:
        self.file_label=self.widgets.HTML(value=f'<b>Downloading</b> {filename}')
        self.progress_bar=self.widgets.IntProgress(
            value=0,
            min=0,
            max=total_size if total_size>0 else 1,
            bar_style='info',
            orientation='horizontal',
            layout=self.widgets.Layout(width='100%' if self.is_colab else '100')
        )
        self.status_label=self.widgets.HTML(value="0%")
        progress_info=self.widgets.HBox([self.progress_bar, self.status_label])
        progress_box=self.widgets.VBox([self.file_label, progress_info])
        self.display(progress_box)

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        self.progress_bar.value = downloaded
        progress_percentage = (downloaded / total_size)*100
        speed = downloaded / elapsed_time if elapsed_time > 0 else 0
                                
        speed_str=f'{speed/(1024**2):.2f} MB/s'
        downloaded_str=DownloadUtils.format_bytes(downloaded)
        total_size_str=DownloadUtils.format_bytes(total_size)
        elapsed_time_str=DownloadUtils.format_time(elapsed_time)
        remaining_time = (total_size - downloaded) / speed if speed > 0 else 0
        remaining_time_str=DownloadUtils.format_time(remaining_time)

        self.status_label.value=(
            f"<b>{progress_percentage:.2f}%</b> ({downloaded_str}/{total_size_str}) "
            f"[{speed_str}, {elapsed_time_str}<{remaining_time_str}]")
        
    def finish(self, time_taken: float)->None:
        self.progress_bar.bar_style='success'
        self.status_label.value=f'<b>Downloaded</b> (Total Time: {DownloadUtils.format_time(time_taken)})'

    def error(self, error_message: str)->None:
        if self.progress_bar:
            self.progress_bar.bar_style='danger'
            self.status_label.value=f'<b>Error</b> {error_message}'

class TqdmProgressHandler(ProgressHandler):
    def __init__(self):
        self.progress_bar=None

    def setup(self, filename: str, total_size: int)->None:
        print(f"downloading: {filename}")
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        self.progress_bar=tqdm(total=total_size, unit='B', unit_scale=True, ncols=None, bar_format=bar_format)

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        self.progress_bar.update(chunk_size)
        speed=downloaded/elapsed_time if elapsed_time > 0 else 0
        speed_str=f'{speed/(1024**2):.2f} MB/s'
        if total_size>0:
            progress_percentage=(downloaded/total_size)*100
            self.progress_bar.set_postfix({
                'percent': f'{progress_percentage:.2f}%',
                'speed': speed_str
            })
        else:
            self.progress_bar.set_postfix({
                'downloaded': DownloadUtils.format_bytes(downloaded),
                'speed': speed_str
            })

    def finish(self, time_taken: float)->None:
        if self.progress_bar:
            self.progress_bar.close()

    def error(self, error_message: str)->None:
        if self.progress_bar:
            self.progress_bar.close()
        print(f'\nError: {error_message}')

class ConsoleProgressHandler(ProgressHandler):
    def __init__(self):
        self.filename=None
        self.total_size_str=None

    def setup(self, filename: str, total_size: int)->None:
        self.filename=filename
        self.total_size_str=DownloadUtils.format_bytes(total_size)

    def update(self, chunk_size: int, downloaded: int, total_size: int, elapsed_time: float)->None:
        speed=downloaded/elapsed_time if elapsed_time>0 else 0
        speed_str=f"{speed/(1024**2):.2f} MB/s"
        if total_size>0:
            progress_percentage=(downloaded/total_size)*100
            downloaded_str = DownloadUtils.format_bytes(downloaded)
            sys.stdout.write(f"\r{self.filename} - {progress_percentage:.2f}% ({downloaded_str} / {self.total_size_str}, {speed_str})")
        else:
            downloaded_str = DownloadUtils.format_bytes(downloaded)
            sys.stdout.write(f"\r{self.filename} - Downloaded: {downloaded_str}, Speed: ({speed_str})")
        sys.stdout.flush()

    def finish(self, time_taken: float)->None:
        time_str=DownloadUtils.format_time(time_taken)
        sys.stdout.write(f'\nDownload completed. File saved as: {self.filename}\n')
        sys.stdout.write(f'Downloaded in {time_str}\n')

    def error(self, error_message: str)->None:
        sys.stdout.write('\n')
        print(f'\nError: {error_message}')

class Downloader:
    CHUNK_SIZE = 1638400
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

    def __init__(self, api_token: str):
        self.api_token = api_token

    def _get_progress_handler(self)->ProgressHandler:
        widgets, _=JupyterEnvironmentDetector.get_ipywidgets()
        is_notebook=JupyterEnvironmentDetector.in_jupyter_notebook()
        is_colab=JupyterEnvironmentDetector.in_colab()
        if widgets and (is_notebook or is_colab):
            return NotebookProgressHandler()
        elif tqdm:
            return TqdmProgressHandler()
        return ConsoleProgressHandler()

    def start_download_thread(self, file: ModelVersionFile, local_dir: str, overwrite: bool=False)->threading.Thread:
        thread=threading.Thread(target=self._download_file, args=(file, local_dir, overwrite))
        thread.start()
        return thread
    
    def _download_file(self, file: ModelVersionFile, save_dir: str, overwrite: bool=False):
        url=file.downloadUrl
        filename=file.name
        filesize=int(float(file.sizeKB)*1024)

        output_file=os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(output_file) and not overwrite:
            print(f"File already exists: {filename}")
            return

        headers={
            'Authorization': f'Bearer {self.api_token}',
            'User-Agent': self.USER_AGENT,
        }
        
        progress_handler=self._get_progress_handler()
        downloaded=0
        start_time=time.time()

        try:
            session=requests.Session()
            response=session.get(url, headers=headers, stream=True)
            response.raise_for_status()

            total_size=int(response.headers.get('content-length', 0))
            if total_size==0 and filesize>0:
                total_size=filesize
            
            progress_handler.setup(filename, total_size)

            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_handler.update(len(chunk), downloaded, total_size, time.time()-start_time)
                    
            progress_handler.finish(time.time()-start_time)

        except Exception as e:
            progress_handler.error(str(e))
    
class DownloadManager:
    def __init__(self, model_info, local_dir, token):
        self.token=token
        self.model_info=model_info
        self.local_dir=local_dir
        self.downloader=Downloader(api_token=token)
        self.api=CivitAIClient(api_token=token)
        self.download_files=[]

    def get_download_files(self):
        model=self.api.get_model(self.model_info.get('id'))
        for version in model.modelVersions:
            for file_data in version.files:
                file=ModelVersionFile(**file_data)
                if file.downloadUrl:
                    self.downloader.start_download_thread(file, self.local_dir)
                    self.download_files.append(file.downloadUrl)
                    return file.downloadUrl, file.name, int(float(file.sizeKB) * 1024), self.local_dir, self.token
                time.sleep(1)
            time.sleep(1)

    def download_all_files(self):
        model=self.api.get_model(self.model_info.get('id'))
        for version in model.modelVersions:
            for file_data in version.files:
                file=ModelVersionFile(**file_data)
                if file.downloadUrl:
                    self.downloader.start_download_thread(file, self.local_dir)
                    self.download_files.append(file.downloadUrl)
                time.sleep(1)

    def version_download_all_files(self, version_id: int):
        model=self.api.get_model(self.model_info.get('id'))
        for version in model.modelVersions:
            if version.id==version_id:
                for file_data in version.files:
                    file=ModelVersionFile(**file_data)
                    if file.downloadUrl:
                        self.downloader.start_download_thread(file, self.local_dir)
                        self.download_files.append(file.downloadUrl)
                    time.sleep(1)