import os.path
import sys
import time
import urllib.request
import threading
from urllib.parse import urlparse, parse_qs, unquote, urljoin

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets=None

CHUNK_SIZE = 1638400
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

base_url="https://civitai.com/api/download/models/"

def in_jupyter_notebook():
    try:
       from IPython import get_ipython
       if 'IPKernelApp' in get_ipython().config:
           return True
    except:
        pass
    return False

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

def download_file(url: str, output_path: str, token: str):
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': USER_AGENT,
    }

    # Disable automatic redirect handling
    class NoRedirection(urllib.request.HTTPErrorProcessor):
        def http_response(self, request, response):
            return response
        https_response = http_response
    try:
        request = urllib.request.Request(url, headers=headers)
        opener = urllib.request.build_opener(NoRedirection)
        response = opener.open(request)

        status_code=response.getcode()
        if status_code in [301, 302, 303, 307, 308]:
            redirect_url = response.getheader('Location')

            # Extract filename from the redirect URL
            parsed_url = urlparse(redirect_url)
            query_params = parse_qs(parsed_url.query)
            content_disposition = query_params.get('response-content-disposition', [None])[0]

            if content_disposition:
                filename = unquote(content_disposition.split('filename=')[1].strip('"'))
            else:
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    raise Exception('Unable to determine filename')

            response = urllib.request.urlopen(redirect_url)
        elif response.status == 404:
            raise Exception('File not found')
        else:
            raise Exception('No redirect found, something went wrong')

        total_size = response.getheader('Content-Length')
        if total_size is not None:
            total_size = int(total_size)
        else:
            total_size=0

        output_file = os.path.join(output_path, filename)
        os.makedirs(output_path, exist_ok=True)

        downloaded = 0
        start_time = time.time()

        is_notebook = in_jupyter_notebook() and widgets

        if is_notebook:
            progress_bar=widgets.IntProgress(
                value=0,
                min=0,
                max=total_size if total_size > 0 else 1,
                description=filename,
                bar_style='info',
                orientatiion='horizontal'
            )
            progress_label=widgets.Label('0 bytes')
            progress_box=widgets.VBox([progress_bar, progress_label])
            display(progress_box)
        elif tqdm:
            progress_bar=tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
        else:
            progress_bar=None

        with open(output_file, 'wb') as f:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    speed = downloaded / elapsed_time
                    speed_str=f'{speed/(1024**2):.2f} MB/s'
                else:
                    speed_str='0.00 MB/s'

                buffer = response.read(CHUNK_SIZE)
                if not buffer:
                    break

                f.write(buffer)
                downloaded += len(buffer)
                if progress_bar:
                    if is_notebook:
                        if total_size > 0:
                            progress_bar.value = downloaded
                            progress_percentage = (downloaded / total_size)*100
                            progress_bar.description=f'{filename}... {progress_percentage:.2f}% - {speed_str}'
                            progress_label.value=f'{downloaded} / {total_size} bytes'
                        else:
                            progress_bar.value=1
                            progress_bar.description=f'{filename}... Downloading'
                            progress_label.value=f'{downloaded} bytes'
                    elif tqdm:
                        progress_bar.update(len(buffer))
        
        end_time = time.time()
        time_taken = end_time - start_time
        hours, remainder = divmod(time_taken, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            time_str = f'{int(hours)}h {int(minutes)}m {int(seconds)}s'
        elif minutes > 0:
            time_str = f'{int(minutes)}m {int(seconds)}s'
        else:
            time_str = f'{int(seconds)}s'

        if progress_bar:
            if is_notebook:
                progress_bar.bar_style='success'
                progress_bar.description=f'{filename}... Downloaded'
                progress_label.value=f'{downloaded} / {total_size} bytes ({time_str})'
            elif tqdm:
                progress_bar.close()
        
        print(f'Download completed. File saved as: {filename}')
        print(f'Downloaded in {time_str}')

    except Exception as e:
        print(f'Error: {e}')
        if progress_bar:
            if is_notebook:
                progress_bar.bar_style='danger'
                progress_bar.description='Error'
        elif tqdm:
            progress_bar.close()