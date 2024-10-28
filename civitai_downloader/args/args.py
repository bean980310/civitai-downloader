import argparse
    
def register():
    parser=argparse.ArgumentParser(description='Download a model from CivitAI')
    parser.add_argument('model_id', type=int, help='CivitAI Model ID, eg: 46846')
    parser.add_argument('local_dir', type=str, help='Output path, eg: /workspace/stable-diffusion-webui/models/Stable-diffusion', default='./')
    parser.add_argument('--version', action='version', version='%(prog)s 0.2.1', help='Show the version number and exit')

    return parser.parse_args()