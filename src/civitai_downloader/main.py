import sys
from civitai_downloader import cli

def main():
    if len(sys.argv)==1:
        sys.argv.append('--help')
    cli.civitai_downloader_cli()