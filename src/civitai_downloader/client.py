from civitai_downloader.api import CivitAIClient

class APIClient(CivitAIClient):
    def __init__(self, api_key: str = None) -> None:
        super().__init__(api_key)