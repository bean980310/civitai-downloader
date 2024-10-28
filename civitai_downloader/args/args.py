import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='CivitAI Downloader',
    )

    parser.add_argument(
        'name', nargs='?', default='civitai-downloader', help='Name of the program'
    )
    parser.add_argument(
        '--version', action='version', version='%(prog)s 0.2.1',
        help='Show the version number and exit'
    )

    parser.add_argument(
        '--model_id',
        dest='model_id',
        type=int,
        help='CivitAI Model ID, eg: 46846',
        required=False if 'url' in locals() else True
    )

    parser.add_argument(
        '--url',
        dest='url',
        type=str,
        help='CivitAI Download URL, eg: https://civitai.com/api/download/models/46846',
        required=False if 'model_id' in locals() else True
    )

    parser.add_argument(
        '--local-dir',
        dest='local_dir',
        type=str,
        help='Output path, eg: /workspace/stable-diffusion-webui/models/Stable-diffusion',
        required=False,
        default='./'
    )

    return parser.parse_args()