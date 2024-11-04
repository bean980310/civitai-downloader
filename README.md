# Model Downloader for CivitAI
Model Downloader Package for Python Developer and CLI for CivitAI.

It is recommanded on Amazon Web Services, Microsoft Azure, Google Cloud Platform.

## How to Use

First, Install civitai-model-downloader

```bash
# Install from PyPI
pip3 install civitai-model-downloader

# Install from GitHub
pip3 install git+https://github.com/bean980310/civitai-downloader.git

# Install from Source
git clone https://github.com/bean980310/civitai-downloader.git
cd civitai-downloader
pip3 install -e .
```

and, Insert your Access token

```python
from civitai_downloader import login

login()
```

Import Your CivitAI API Token and Next, Download a model

```python
from civitai_downloader import login
from civitai_downloader import civitai_download, url_download

login()

# example
url_download(url="https://civitai.com/api/download/models/90854", local_dir="./models/checkpoints/sd15")

# or
civitai_download(model_version_id=90854, local_dir="./models/checkpoints/sd15")
```

Also, you can use to civitai-downloader command line

```bash
# example
civitai-downloader-cli download 90854

# prefix local dir
civitai-downloader-cli download 90854 --local-dir ./models/checkpoints/sd15

# to use url
civitai-downloader-cli url-download https://civitai.com/api/download/models/90854
```