from typing import Dict, Optional

CIVITAI_API_URL='https://civitai.com/api/v1'

class BaseAPI:
    def __init__(self, api_token: Optional[str]=None, api_url: Optional[str]=None):
        self.api_token = api_token
        self.api_url = api_url or CIVITAI_API_URL
    
    def _get_headers(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {self.api_token}'} if self.api_token is not None else {}