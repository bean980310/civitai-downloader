import requests

from civitai_downloader.api.base import BaseAPI
from civitai_downloader.api_class import Model, ModelVersionFileMetadata, ModelVersionFile, ModelVersionImages, ModelVersions

class ModelAPI(BaseAPI):
    def get_model_info_from_api(self, model_id: int)->Model:
        api_url=f'{self.api_url}/models/{model_id}'
        headers=self._get_headers()
        response=requests.get(api_url, headers=headers)
        if response.status_code==200:
            data=response.json()
            return self._parse_model(data)
        else:
            response.raise_for_status()

    def _parse_model(self, data: dict)->Model:
        modelVersions=[]
        for versions_data in data.get('modelVersions'):
            files=[]
            for file_data in versions_data.get('files'):
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
                    downloadUrl=file_data.get('downloadUrl'),
                    primary=file_data.get('primary',False),
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
            version=ModelVersions(
                id=versions_data.get('id'),
                modelId=versions_data.get('modelId'),
                name=versions_data.get('name'),
                createdAt=versions_data.get('createdAt'),
                updatedAt=versions_data.get('updatedAt'),
                trainedWords=versions_data.get('trainedWords',[]),
                baseModel=versions_data.get('baseModel'),
                earlyAccessTimeFrame=versions_data.get('earlyAccessTimeFrame'),
                description=versions_data.get('description'),
                stats=versions_data.get('stats', {}),
                files=files,
                images=images,
                downloadUrl=versions_data.get('downloadUrl'),
            )
            modelVersions.append(version)
        model=Model(
            id=data.get('id'),
            name=data.get('name'),
            description=data.get('description'),
            type=data.get('type'),
            poi=data.get('poi'),
            nsfw=data.get('nsfw'),
            allowNoCredit=data.get('allowNoCredit'),
            allowCommercialUse=data.get('allowCommercialUse'),
            allowDerivates=data.get('allowDerivates'),
            allowDifferentLicense=data.get('allowDifferentLicense'),
            stats=data.get('stats', {}),
            creator=data.get('creator', {}),
            tags=data.get('tags', []),
            modelVersions=modelVersions,
            mode=data.get('mode')
        )
        return model