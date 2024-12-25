# manager.py
import os
import threading
from typing import List

from civitai_downloader.api.client import CivitAIClient
from civitai_downloader.api_class import ModelVersionFile
from civitai_downloader.download.handler import DownloadHandler

class DownloadManager:
    def __init__(self, model, local_dir: str, token: str):
        self.model = model
        self.local_dir = local_dir
        self.token = token
        self.api = CivitAIClient(api_token=token)
        self.threads = []

    def download_all_files(self):
        """
        예) model에 있는 모든 modelVersion/files를 불러와
        각각 스레드로 다운로드
        """
        # 예시로 model.modelVersions의 모든 파일을 가져온다고 가정
        for version in self.model.modelVersions:
            for file_data in version.files:
                file_obj = ModelVersionFile(**file_data)
                if file_obj.downloadUrl:
                    t = threading.Thread(
                        target=self._download_single_file,
                        args=(file_obj,)
                    )
                    t.start()
                    self.threads.append(t)

        # === 스레드 모두 끝날 때까지 대기 ===
        for t in self.threads:
            t.join()

        print("모든 파일 다운로드 완료!")

    def version_download_all_files(self, version_id: int):
        """특정 버전 ID만 다운로드"""
        # 예시
        version = self.api.get_model_version(version_id)
        if not version:
            print(f"No model version found for {version_id}")
            return

        for file_data in version.files:
            file_obj = ModelVersionFile(**file_data)
            if file_obj.downloadUrl:
                t = threading.Thread(
                    target=self._download_single_file,
                    args=(file_obj,)
                )
                t.start()
                self.threads.append(t)

        # join
        for t in self.threads:
            t.join()

        print(f"version {version_id} file downloaded!")

    def _download_single_file(self, file_obj: ModelVersionFile):
        """
        스레드용 내부 함수: DownloadHandler 통해 단일 파일 다운로드
        """
        handler = DownloadHandler(self.token)
        handler.process_download([file_obj], self.local_dir)