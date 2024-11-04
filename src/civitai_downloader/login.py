from civitai_downloader.token import create_token_manager

def login():
    token_manager = create_token_manager()
    token_manager.login()