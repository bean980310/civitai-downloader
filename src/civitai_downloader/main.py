import sys
from civitai_downloader.cli import CivitaiDownloaderCLI

def main():
    if len(sys.argv)==1:
        sys.argv.append('--help')
    cli = CivitaiDownloaderCLI()
    cli.run()