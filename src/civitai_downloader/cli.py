import typer
from civitai_downloader.download.download import civitai_download, url_download
from civitai_downloader.token.token import get_token, prompt_for_civitai_token

__all__=[
    'civitai_downloader_cli',
]

civitai_downloader_cli=typer.Typer()

@civitai_downloader_cli.command("download", help="Download models from CivitAI")
def civitai_download_cmd(model_ver_id: str=typer.Argument(..., help="Model version ID"),
                      local_dir: str=typer.Option(".", help="Output path")):
    civitai_download(model_id=model_ver_id, local_dir=local_dir, token=get_token())
    return model_ver_id, local_dir

@civitai_downloader_cli.command("url-download", help="Download models from CivitAI with URL")
def url_download_cmd(model_url: str=typer.Argument(..., help="Model URL"),
                      local_dir: str=typer.Option(".", help="Output path")):
    url_download(url=model_url, local_dir=local_dir, token=get_token())
    return model_url, local_dir

@civitai_downloader_cli.command("token", help="Store CivitAI API token")
def token_cmd():
    prompt_for_civitai_token()