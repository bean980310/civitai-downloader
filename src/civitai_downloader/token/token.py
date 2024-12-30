# src/civitai_downloader/token/token.py

import os
import requests
import getpass
from pathlib import Path
from typing import Optional

TOKEN_FILE = Path.home() / '.civitai' / 'config'
CIVITAI_HOST = 'https://civitai.com'
ENV_TOKEN_NAME = 'CIVITAI_TOKEN'

class TokenManager:
    """
    CivitAI 다운로드를 위한 토큰을 관리하는 클래스.
    1) 환경 변수 (CIVITAI_TOKEN)
    2) 로컬 설정 파일 (~/.civitai/config)
    3) 사용자 입력 (prompt)
    순으로 우선순위를 두고 조회한다.
    """
    def __init__(self, host: str = CIVITAI_HOST):
        self.host = host

    def get_token(self, prompt_if_missing: bool = True) -> Optional[str]:
        """
        토큰을 가져온다:
          1) ENV_TOKEN_NAME 환경 변수
          2) TOKEN_FILE 파일
          3) (옵션) 사용자 입력 프롬프트
        순으로 찾고, 찾으면 검증(선택) 후 반환.
        """
        # 1) 환경 변수
        token = os.environ.get(ENV_TOKEN_NAME)
        if token:
           return token

        # 2) 로컬 설정 파일
        if TOKEN_FILE.is_file():
            try:
                file_token = TOKEN_FILE.read_text().strip()
                return file_token
            except Exception as e:
                print(f"[ERROR] Failed to read token file: {e}")

        # 3) 프롬프트로 입력받기
        if prompt_if_missing:
            new_token = self.prompt_for_token()
            if new_token:
                self.store_token(new_token)
                return new_token

        return None

    def prompt_for_token(self) -> Optional[str]:
        """
        사용자에게 토큰 입력을 요청한다.
        주피터/코랩 환경인지 여부에 따라 input() 또는 widgets.Text() 등을 쓰도록 조정 가능.
        """
        try:
            token = getpass.getpass("Enter CivitAI API Token: ").strip()
            return token if token else None
        except EOFError:
            return None

    def store_token(self, token: str) -> None:
        """
        로컬 설정 파일에 토큰을 저장한다.
        """
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(token.strip(), encoding='utf-8')
        print(f"Token file saved to {TOKEN_FILE}.")