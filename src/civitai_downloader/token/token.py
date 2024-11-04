from getpass import getpass
from pathlib import Path
import threading
from typing import Optional, Tuple
from abc import ABC, abstractmethod

class TokenManager:
    def __init__(self, token_dir: str='.civitai'):
        self.token_file=Path.home()/token_dir/'config'
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

    def get_token(self)->Optional[str]:
        try:
            with open(self.token_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            return None
        
    def store_token(self, token: str)->None:
        with open(self.token_file, 'w') as f:
            f.write(token)

    def login(self)->str:
        existing_token=self.get_token()
        if existing_token:
            print('CivitAI API token already exists.')
            return existing_token
        return self._prompt_for_token()
    
    @abstractmethod
    def _prompt_for_token(self)->str:
        pass

class ConsoleTokenManager(TokenManager):
    def _prompt_for_token(self)->str:
        try:
            token = getpass('Please enter your CivitAI API token: ')
        except Exception as e:
            token = input('Please enter your CivitAI API token: ')
        self.store_token(token)
        return token
    
class NotebookTokenManager(TokenManager):
    def __init__(self, token_dir: str='.civitai'):
        super().__init__(token_dir)
        from civitai_downloader.env import JupyterEnvironmentDetector
        self.env_detector=JupyterEnvironmentDetector
        self.widgets, self.display=self.env_detector.get_ipywidgets()
        self.is_colab=self.env_detector.in_colab()

    def _prompt_for_token(self)->str:
        token_widget=self.widgets.Password(
            description='CivitAI API Token:',
            placeholder='Enter your CivitAI API token',
            style={'description_width': 'initial'},
            layout=self.widgets.Layout(width='50%' if self.is_colab else 'auto')
        )

        submit_button=self.widgets.Button(
            description='Submit',
            layout=self.widgets.Layout(width='100px'))
        
        output=self.widgets.Output()

        container=self.widgets.VBox([
            token_widget,
            submit_button,
            output
        ], layout=self.widgets.Layout(
            padding='10px',
            align_items='flex-start'
        ))

        token_event=threading.Event()
        token=None

        def on_submit(b):
            nonlocal token
            token=token_widget.value
            self.store_token(token)
            with output:
                print('Token stored successfully.')
            container.close()
            token_event.set()
            
        submit_button.on_click(on_submit)
        self.display(container)
            
        token_event.wait()
        return token

def create_token_manager(token_dir: str='.civitai')->TokenManager:
    try:
        from civitai_downloader.env import JupyterEnvironmentDetector
        is_notebook=JupyterEnvironmentDetector.in_jupyter_notebook()
        is_colab=JupyterEnvironmentDetector.in_colab()
        widgets, _=JupyterEnvironmentDetector.get_ipywidgets()

        if widgets and (is_notebook or is_colab):
            return NotebookTokenManager(token_dir)
    except ImportError:
        pass

    return ConsoleTokenManager(token_dir)