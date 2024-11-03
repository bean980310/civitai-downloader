from typing import Optional
import requests

from civitai_downloader.api.model import ModelAPI
from civitai_downloader.api_class import Sort, Period, ModelType, AllowCommercialUse, BaseModel, ModelList, Metadata


class ModelsAPI(ModelAPI):
    def list_models(self, 
                    limit: Optional[int]=100, 
                    page: Optional[int]=1,
                    query: Optional[str]=None,
                    tag: Optional[str]=None,
                    username: Optional[str]=None,
                    types: Optional[ModelType]=None,
                    sort: Optional[Sort]="Highest Rated",
                    period: Optional[Period]="Week",
                    favorites: Optional[bool]=None,
                    hidden: Optional[bool]=None,
                    primaryFileOnly: Optional[bool]=None,
                    allowNoCredit: Optional[bool]=None,
                    allowDerivates: Optional[bool]=None,
                    allowDifferentLicenses: Optional[bool]=None,
                    allowCommercialUse: Optional[AllowCommercialUse]=None,
                    baseModel: Optional[BaseModel]=None,
                    nsfw: Optional[bool]=None,
                    supportsGeneration: Optional[bool]=None)->ModelList:
        api_url=f'{self.api_url}/models'
        headers=self._get_headers()
        params = self._construct_params(locals())

        response=requests.get(api_url, params=params, headers=headers)
        if response.status_code==200:
            data=response.json()
            models=[]
            for model_data in data.get('items', []):
                model=self._parse_model(model_data)
                models.append(model)

            metadata_data=data.get('metadata', {})
            metadata=Metadata(
                totalItems=metadata_data.get('totalItems', 0),
                currentPage=metadata_data.get('currentPage', 1),
                pageSize=metadata_data.get('pageSize', len(models)),
                totalPages=metadata_data.get('totalPages', 1),
                nextPage=metadata_data.get('nextPage'),
                prevPage=metadata_data.get('prevPage')
            )
            model_list=ModelList(items=models, metadata=metadata)
            return model_list
        else:
            response.raise_for_status()
        
    def _construct_params(self, kwargs):
        params={
            'limit': kwargs.get('limit'),
            'page': kwargs.get('page'),
            'query': kwargs.get('query'),
            'tag': kwargs.get('tag'),
            'username': kwargs.get('username'),
            'types': [t.value for t in kwargs.get('types', [])] if kwargs.get('types') else None,
            'sort': kwargs.get('sort').value if kwargs.get('sort') else None,
            'period': kwargs.get('period').value if kwargs.get('period') else None,
            'favorites': kwargs.get('favorites'),
            'hidden': kwargs.get('hidden'),
            'primaryFileOnly': kwargs.get('primaryFileOnly'),
            'allowNoCredit': kwargs.get('allowNoCredit'),
            'allowDerivates': kwargs.get('allowDerivates'),
            'allowDifferentLicenses': kwargs.get('allowDifferentLicenses'),
            'baseModel': [m.value for m in kwargs.get('baseModel', [])] if kwargs.get('baseModel') else None,
            'nsfw': kwargs.get('nsfw'),
            'supportGeneration': kwargs.get('supportGeneration')
        }

        if kwargs.get('types'): params['types']=','.join([t.value for t in kwargs['types']])
        if kwargs.get('baseModel'): params['baseModel']=','.join([m.value for m in kwargs['baseModel']])
        if kwargs.get('allowCommercialUse'): params['allowCommercialUse']=','.join([acu.value for acu in kwargs['allowCommercialUse']])

        return {k: v for k, v in params.items() if v is not None}
    