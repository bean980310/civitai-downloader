# filename_extractor.py

from urllib.parse import urlparse, parse_qs, unquote

class FileNameExtractor:
    def __init__(self, redirect_url: str):
        self.redirect_url = redirect_url

    def extract_filename(self) -> str:
        parsed_url = urlparse(self.redirect_url)
        query_params = parse_qs(parsed_url.query)
        content_disposition = query_params.get('response-content-disposition', [None])[0]

        if not content_disposition:
            raise Exception('Unable to determine filename')

        # 예: content_disposition = 'attachment; filename="example.safetensors"'
        if 'filename=' in content_disposition:
            filename_part = content_disposition.split('filename=')[1].strip('"')
            return unquote(filename_part)  # URL 디코딩
        else:
            raise Exception('filename= 정보가 존재하지 않습니다.')