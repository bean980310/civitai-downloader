import os.path
import sys
import time
import urllib.request
from urllib.parse import urlparse, parse_qs, unquote
from civitai_downloader.download.format import format_bytes, format_time
import requests

CHUNK_SIZE = 1638400
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets=None

def in_jupyter_notebook():
    try:
       from IPython import get_ipython
       if 'IPKernelApp' in get_ipython().config:
           return True
    except:
        pass
    return False

def download_file(url: str, filename: str, filesize: float, output_path: str, token: str):
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': USER_AGENT,
    }
    try:
        output_file = os.path.join(output_path, filename)
        os.makedirs(output_path, exist_ok=True)

        downloaded = 0
        start_time = time.time()

        session=requests.Session()
        response=session.get(url, headers=headers, stream=True)
        if response.status_code!=200:
            print(f"Download Failed!: {filename}")
            return
        
        total_size=int(response.headers.get('content-length', 0))
        if total_size==0 and filesize>0:
            total_size=int(filesize)
        total_size_str=format_bytes(total_size) if total_size>0 else 'Unknown'

        is_notebook = in_jupyter_notebook() and widgets

        if is_notebook:
            file_label=widgets.HTML(value=f'<b>Downloading</b> {filename}')
            progress_bar=widgets.IntProgress(
                value=0,
                min=0,
                max=total_size if total_size > 0 else 1,
                bar_style='info',
                orientatiion='horizontal',
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
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
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
                            downloaded_str=format_bytes(downloaded)
                            elapsed_time_str=format_time(elapsed_time)
                            remaining_time = (total_size - downloaded) / speed if speed > 0 else 0
                            remaining_time_str=format_time(remaining_time)
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
                                    'downloaded': format_bytes(downloaded),
                                    'speed': speed_str
                                    })
                    else:
                        elapsed_time=time.time()-start_time
                        speed=downloaded/elapsed_time if elapsed_time>0 else 0
                        speed_str=f"{speed/(1024**2):.2f} MB/s"
                        if total_size>0:
                            progress_percentage=(downloaded/total_size)*100
                            downloaded_str = format_bytes(downloaded)
                            sys.stdout.write(f"\r{filename} - {progress_percentage:.2f}% ({downloaded_str} / {total_size_str}, {speed_str})")
                        else:
                            downloaded_str=format_bytes(downloaded)
                            sys.stdout.write(f"\r{filename} - Downloaded: {downloaded_str}, Speed: ({speed_str})")
                        sys.stdout.flush()
        
        end_time = time.time()
        time_taken = end_time - start_time
        time_str=format_time(time_taken)

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