# civitai-downloader

## How to Use

First, Install civitai-downloader

```bash
pip3 install git+https://github.com/bean980310/civitai-downloader.git
```

and, Insert your Access token

```python
from civitai_downloader.token.token import prompt_for_civitai_token

prompt_for_civitai_token()
```

Import Your CivitAI API Token and Next, Download a model

```python
from civitai_downloader.token.token import get_token
from civitai_downloader.download.download import download_file, civitai_download

token=get_token()

# example
download_file(url="https://civitai.com/api/download/models/90854", output_path="./models/checkpoints/sd15", token=token)

# or
civitai_download(model_id=90854, local_dir="./models/checkpoints/sd15", token=token)
```

Also, you can use to civitai-downloader command line

```bash
# example
civitai-downloader 90854 ./models/checkpoints/sd15
```