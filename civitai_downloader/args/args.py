import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='CivitAI Downloader',
    )

    parser.add_argument(
        '--help',
        default=True,
        action='help'
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        'model_id',
        type=int,
        help='CivitAI Model ID, eg: 46846'
    )
    group.add_argument(
        'url',
        type=str,
        help='CivitAI Download URL, eg: https://civitai.com/api/download/models/46846'
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='Output path, eg: /workspace/stable-diffusion-webui/models/Stable-diffusion'
    )

    return parser.parse_args()