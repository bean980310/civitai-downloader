from getpass import getpass
from pathlib import Path
import threading
from typing import Optional, Tuple

from civitai_downloader.env import JupyterEnvironmentDetector

TOKEN_FILE = Path.home() / '.civitai' / 'config'

def get_token()->Optional[str]:
    try:
        with open(TOKEN_FILE, 'r') as file:
            token = file.read()
            return token
    except Exception as e:
        return None


def store_token(token: str)->None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TOKEN_FILE, 'w') as file:
        file.write(token)

def prompt_for_token_base()->str:
    try:
        token = getpass('Please enter your CivitAI API token: ')
    except Exception as e:
        token = input('Please enter your CivitAI API token: ')
    store_token(token)
    return token

def prompt_for_token_notebook()->str:
    widgets, display=JupyterEnvironmentDetector.get_ipywidgets()
    is_colab=JupyterEnvironmentDetector.in_colab()

    token_widget=widgets.Password(
        description='CivitAI API Token:',
        placeholder='Enter your CivitAI API token',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%' if is_colab else 'auto')
    )

    submit_button=widgets.Button(
        description='Submit',
        layout=widgets.Layout(width='100px'))
    
    output=widgets.Output()

    container=widgets.VBox([
        token_widget,
        submit_button,
        output
    ], layout=widgets.Layout(
        padding='10px',
        align_items='flex-start'
    ))
    display(container)

    token_event=threading.Event()
    token=None

    def on_submit(b):
        nonlocal token
        token=token_widget.value
        store_token(token)
        with output:
            print('Token stored successfully.')
        container.close()
        token_event.set()
        
    submit_button.on_click(on_submit)
    display(container)
        
    token_event.wait()
    return token

def prompt_for_civitai_token()->str:
    existing_token=get_token()
    if existing_token:
        print('CivitAI API token already exists.')
        return existing_token
    
    is_notebook=JupyterEnvironmentDetector.in_jupyter_notebook()
    is_colab=JupyterEnvironmentDetector.in_colab()
    widgets, _ = JupyterEnvironmentDetector.get_ipywidgets()

    if widgets and (is_notebook or is_colab):
        return prompt_for_token_notebook()
    
    return prompt_for_token_base()