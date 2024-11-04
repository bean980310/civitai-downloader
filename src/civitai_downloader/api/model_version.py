import requests

from civitai_downloader.api.base import BaseAPI
from civitai_downloader.api_class import ModelVersion, ModelVersionFile, ModelVersionFileMetadata, ModelVersionImages

class ModelVersionAPI(BaseAPI):
    def get_model_version_info_from_api(self, model_version_id: int)->ModelVersion:
        api_url=f'{self.api_url}/model-versions/{model_version_id}'
        headers=self._get_headers()
        response=requests.get(api_url, headers=headers)
        if response.status_code==200:
            data=response.json()
            return self._parse_model_version(data)
        else:
            response.raise_for_status()

    def get_model_version_info_by_hash_from_api(self, hash: str)->ModelVersion:
        api_url=f'{self.api_url}/model-versions/by-hash/{hash}'
        headers={'Authorization': f'Bearer {self.api_token}'} if self.api_token is not None else {}
        response=requests.get(api_url, headers=headers)
        if response.status_code==200:
            data=response.json()
            return self._parse_model_version(data)
        else:
            response.raise_for_status()
            
    def _parse_model_version(self, data: dict)->ModelVersion:
        files=[]
        for file_data in data.get('files', []):
            metadata_data=file_data.get('metadata',{})
            metadata=ModelVersionFileMetadata(
                fp=metadata_data.get('fp'),
                size=metadata_data.get('size'),
                format=metadata_data.get('format'),
            )
            file=ModelVersionFile(
                name=file_data.get('name'),
                id=file_data.get('id'),
                sizeKB=file_data.get('sizeKB',0),
                type=file_data.get('type'),
                metadata=metadata,
                pickleScanResult=file_data.get('pickleScanResult'),
                pickleScanMessage=file_data.get('pickleScanMessage'),
                virusScanResult=file_data.get('virusScanResult'),
                scannedAt=file_data.get('scannedAt'),
                hashes=file_data.get('hashes',{}),
                primary=file_data.get('primary',False),
                downloadUrl=file_data.get('downloadUrl'),
            )
            files.append(file)

        images=[]
        for image_data in data.get('images',[]):
            image=ModelVersionImages(
                url=image_data.get('url'),
                nsfw=image_data.get('nsfw'),
                width=image_data.get('width'),
                height=image_data.get('height'),
                hash=image_data.get('hash'),
                meta=image_data.get('meta',{}),
            )
            images.append(image)

        model_version=ModelVersion(
            id=data.get('id'),
            modelId=data.get('modelId'),
            name=data.get('name'),
            createdAt=data.get('createdAt'),
            updatedAt=data.get('updatedAt'),
            trainedWords=data.get('trainedWords',[]),
            baseModel=data.get('baseModel'),
            earlyAccessTimeFrame=data.get('earlyAccessTimeFrame'),
            description=data.get('description'),
            stats=data.get('stats', {}),
            model=data.get('model', {}),
            files=files,
            images=images,
            downloadUrl=data.get('downloadUrl'),
        )
        return model_version