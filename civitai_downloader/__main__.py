from download.download import download_file
from token.token import get_token, store_token, prompt_for_civitai_token

if __name__=='__main__':
    token = get_token()

    if token is None:
        token = prompt_for_civitai_token()