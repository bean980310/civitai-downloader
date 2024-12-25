from civitai_downloader.token.token import TokenManager

def login() -> str:
    """
    다운로드 시 필요한 토큰을 가져옴.
    """
    manager = TokenManager()
    token = manager.get_token(prompt_if_missing=True)
    if not token:
        raise RuntimeError("Failed to load CivitAI API token!")
    return token