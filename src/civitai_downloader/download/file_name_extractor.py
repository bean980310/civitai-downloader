import re
from urllib.parse import urlparse, parse_qs, unquote
from typing import Optional

class FileNameExtractor:
    @staticmethod
    def from_url(url: str) -> Optional[str]:
        """
        URL 쿼리 파라미터(예: 'response-content-disposition')에서
        파일명을 추출해 반환. 못 찾으면 None.
        """
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)

        content_disp = query_params.get('response-content-disposition', [None])[0]
        if not content_disp:
            return None
        
        match = re.search(r'filename="?([^"]+)"?', content_disp)
        if match:
            return unquote(match.group(1))
        return None