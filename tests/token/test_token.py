from civitai_downloader import login

def test_token():
    token=login()
    assert token