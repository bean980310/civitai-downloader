from civitai_downloader.token.token import TokenManager
from typing import Optional

def login(
        token: Optional[str] = None,
) -> str:
    """
    다운로드 시 필요한 토큰을 가져옴.
    """
    manager = TokenManager()
    if token is None:
        token = manager.get_token(prompt_if_missing=True)
    if not token:
        raise RuntimeError("Failed to load CivitAI API token!")
    return token