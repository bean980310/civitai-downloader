from civitai_downloader.api.creators import get_creators_info_from_api

def test_get_creators_info_from_api():
    creators = get_creators_info_from_api()
    assert len(creators) > 0
    assert "username" in creators[0]
    assert "modelCount" in creators[0]
    assert "link" in creators[0]