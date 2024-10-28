from civitai_downloader.download.download import civitai_download
from civitai_downloader.token.token import get_token, prompt_for_civitai_token
from civitai_downloader.args.args import register
from argparse import ArgumentParser

def main():

    ArgumentParser('civitai-downloader', usage="civitai-downloader <model_id> [local_dir]")
    args=register()

    model_id=args.model_id
    local_dir=args.local_dir
    token=get_token()

    if not token:
        token=prompt_for_civitai_token()

    try:
        civitai_download(model_id=model_id,local_dir=local_dir,token=token)
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()