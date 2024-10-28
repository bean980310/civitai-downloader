from civitai_downloader.download.download import civitai_download, url_download
from civitai_downloader.token.token import get_token, prompt_for_civitai_token
from civitai_downloader.args.args import get_args

def main():
    args=get_args()
    if args.model_id:
        src=args.model_id
    else:
        src=args.url
    local_dir=args.output_path
    token=get_token()

    if not token:
        token=prompt_for_civitai_token()

    try:
        if args.model_id:
            civitai_download(model_id=src,local_dir=local_dir,token=token)
        else:
            url_download(url=src,local_dir=local_dir,token=token)
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()