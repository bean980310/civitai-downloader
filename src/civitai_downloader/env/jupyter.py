
class JupyterEnvironmentDetector:
    """A class to detect and handle Jupyter notebook environment.
    
    This class provides methods to check if code is running in a Jupyter notebook
    environment and handles related widget functionality.
    """
    
    @staticmethod
    def in_jupyter_notebook():
        """Detect if the code is running in a Jupyter Notebook using IPython configuration."""
        try:
            from IPython import get_ipython
            if 'IPKernelApp' in get_ipython().config:
                return True
        except Exception:
            pass
        return False
    
    @staticmethod
    def in_jupyter_shell():
        """Alternative method to check if running in a Jupyter environment using shell type."""
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter Notebook or Jupyter QtConsole
        except NameError:
            pass
        return False
    
    @staticmethod
    def get_ipywidgets():
        """
        Check if ipywidgets are available in the current environment.
        Returns ipywidgets and display functions if available, else returns None.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
            return widgets, display
        except ImportError:
            return None, None