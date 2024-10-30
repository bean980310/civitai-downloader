from getpass import getpass
from pathlib import Path
import threading

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

def in_jupyter():
    try:
        from IPython import get_ipython
        shell=get_ipython().__class__.__name__
        if shell=='ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False

def prompt_for_civitai_token():
    if in_jupyter():
        import ipywidgets as widgets
        from IPython.display import display

        token_widget=widgets.Password(
            description='CivitAI API Token:',
            placeholder='Enter your CivitAI API token',
            style={'description_width': 'initial'}
        )

        submit_button=widgets.Button(description='Submit')
        output=widgets.Output()
        display(token_widget, submit_button, output)

        token_event=threading.Event()
        token=None

        def on_submit(b):
            nonlocal token
            token=token_widget.value
            store_token(token)
            with output:
                print('Token stored successfully.')
            token_widget.close()
            submit_button.close()
            token_event.set()
        
        submit_button.on_click(on_submit)
        token_event.wait()
        return token
    else:
        try:
            token = getpass('Please enter your CivitAI API token: ')
        except Exception as e:
            token = input('Please enter your CivitAI API token: ')
        store_token(token)
        return token