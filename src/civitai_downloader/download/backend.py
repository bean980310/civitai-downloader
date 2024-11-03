import os.path
import sys
import time
from civitai_downloader.api_class import ModelVersionFile
from civitai_downloader.api import ModelVersionAPI, ModelAPI
import requests
import threading

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets=None

class Downloader:
    CHUNK_SIZE = 1638400
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

    def __init__(self, api_token: str):
        self.api_token = api_token

    def start_download_thread(self, file: ModelVersionFile, local_dir: str, overwrite: bool=False):
        thread=threading.Thread(target=self._download_file, args=(file, local_dir, overwrite))
        thread.start()
        return thread
    
    def _download_file(self, file: ModelVersionFile, save_dir: str, overwrite: bool=False):
        url=file.downloadUrl
        filename=file.name
        filesize=int(float(file.sizeKB)*1024)

        headers={
            'Authorization': f'Bearer {self.api_token}',
            'User-Agent': self.USER_AGENT,
        }
        
        output_file=os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(output_file) and not overwrite:
            print(f"File already exists: {filename}")
            return
        
        downloaded=0
        start_time=time.time()

        try:
            session=requests.Session()
            response=session.get(url, headers=headers, stream=True)
            response.raise_for_status()

            total_size=int(response.headers.get('content-length', 0))
            if total_size==0 and filesize>0:
                total_size=filesize
            total_size_str=self.format_bytes(total_size) if total_size>0 else 'Unknown'

            is_notebook=self.in_jupyter_notebook() and widgets
            if is_notebook:
                file_label=widgets.HTML(value=f'<b>Downloading</b> {filename}')
                progress_bar=widgets.IntProgress(
                    value=0,
                    min=0,
                    max=total_size if total_size>0 else 1,
                    bar_style='info',
                    orientation='horizontal',
                    layout=widgets.Layout(width='100')
                )
                status_label=widgets.HTML(value="0%")
                progress_info=widgets.HBox([progress_bar, status_label])
                progress_box=widgets.VBox([file_label, progress_info])
                display(progress_box)
            elif tqdm:
                print(f"downloading: {filename}")
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                progress_bar=tqdm(total=total_size, unit='B', unit_scale=True, ncols=None, bar_format=bar_format)
            else:
                progress_bar=None

            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                    
                        if progress_bar:
                            if is_notebook:
                                progress_bar.value = downloaded
                                progress_percentage = (downloaded / total_size)*100
                                elapsed_time = time.time() - start_time
                                speed = downloaded / elapsed_time if elapsed_time > 0 else 0
                                speed_str=f'{speed/(1024**2):.2f} MB/s'
                                downloaded_str=self.format_bytes(downloaded)
                                elapsed_time_str=self.format_time(elapsed_time)
                                remaining_time = (total_size - downloaded) / speed if speed > 0 else 0
                                remaining_time_str=self.format_time(remaining_time)
                                status_label.value=(
                                    f"<b>{progress_percentage:.2f}%</b> ({downloaded_str}/{total_size_str}) "
                                    f"[{speed_str}, {elapsed_time_str}<{remaining_time_str}]")
                            elif tqdm:
                                progress_bar.update(len(chunk))
                                speed=downloaded/(time.time()-start_time) if time.time()-start_time > 0 else 0
                                speed_str=f'{speed/(1024**2):.2f} MB/s'
                                if total_size>0:
                                    progress_percentage=(downloaded/total_size)*100
                                    progress_bar.set_postfix({
                                        'percent': f'{progress_percentage:.2f}%',
                                        'speed': speed_str
                                        })
                                else:
                                    progress_bar.set_postfix({
                                        'downloaded': self.format_bytes(downloaded),
                                        'speed': speed_str
                                        })
                        else:
                            elapsed_time=time.time()-start_time
                            speed=downloaded/elapsed_time if elapsed_time>0 else 0
                            speed_str=f"{speed/(1024**2):.2f} MB/s"
                            if total_size>0:
                                progress_percentage=(downloaded/total_size)*100
                                downloaded_str = self.format_bytes(downloaded)
                                sys.stdout.write(f"\r{filename} - {progress_percentage:.2f}% ({downloaded_str} / {total_size_str}, {speed_str})")
                            else:
                                downloaded_str=self.format_bytes(downloaded)
                                sys.stdout.write(f"\r{filename} - Downloaded: {downloaded_str}, Speed: ({speed_str})")
                            sys.stdout.flush()
        
            end_time = time.time()
            time_taken = end_time - start_time
            time_str=self.format_time(time_taken)

            if progress_bar:
                if is_notebook:
                    progress_bar.bar_style='success'
                    status_label.value=f'<b>Downloaded</b> (Total Time: {time_str})'
                elif tqdm:
                    progress_bar.close()
            else:
                sys.stdout.write(f'\nDownload completed. File saved as: {filename}\n')
                sys.stdout.write(f'Downloaded in {time_str}\n')
        except Exception as e:
            print(f'Error: {e}')
            if progress_bar:
                if is_notebook:
                    progress_bar.bar_style='danger'
                    progress_bar.description='Error'
                elif tqdm:
                    progress_bar.close()
            else:
                sys.stdout.write('\n')

    @staticmethod
    def in_jupyter_notebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' in get_ipython().config:
                return True
        except:
            pass
        return False
    
    @staticmethod
    def format_bytes(size):
        for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if size<1024 or unit == 'PB':
                return f"{size:.2f} {unit}"
            size/=1024

    @staticmethod
    def format_time(seconds):
        h, rem=divmod(int(seconds), 3600)
        m, s=divmod(rem, 60)
        result=''
        if h>0:
            result+=f'{int(h)}h '
        if m>0:
            result+=f'{int(m)}m '
        result+=f'{int(s)}s'
        return result.strip()
    
class DownloadManager:
    def __init__(self, model_info, local_dir, token):
        self.token=token
        self.model_info=model_info
        self.local_dir=local_dir
        self.downloader=Downloader(api_token=token)
        self.api=ModelAPI(api_token=token)
        self.api2=ModelVersionAPI(api_token=token)
        self.download_files=[]

    def get_download_files(self):
        model=self.api.get_model_info_from_api(self.model_info.get('id'))
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
        model=self.api.get_model_info_from_api(self.model_info.get('id'))
        for version in model.modelVersions:
            for file_data in version.files:
                file=ModelVersionFile(**file_data)
                if file.downloadUrl:
                    self.downloader.start_download_thread(file, self.local_dir)
                    self.download_files.append(file.downloadUrl)
                time.sleep(1)

    def version_download_all_files(self, version_id: int):
        model=self.api.get_model_info_from_api(self.model_info.get('id'))
        for version in model.modelVersions:
            if version.id==version_id:
                for file_data in version.files:
                    file=ModelVersionFile(**file_data)
                    if file.downloadUrl:
                        self.downloader.start_download_thread(file, self.local_dir)
                        self.download_files.append(file.downloadUrl)
                    time.sleep(1)