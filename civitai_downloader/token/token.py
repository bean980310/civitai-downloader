from pathlib import Path

TOKEN_FILE = Path.home() / '.civitai' / 'config'

def get_token():
    try:
        with open(TOKEN_FILE, 'r') as file:
            token = file.read()
            return token
    except Exception as e:
        return None


def store_token(token: str):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TOKEN_FILE, 'w') as file:
        file.write(token)


def prompt_for_civitai_token():
    token = input('Please enter your CivitAI API token: ')
    store_token(token)
    return token