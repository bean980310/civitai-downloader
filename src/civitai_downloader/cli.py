import argparse
from civitai_downloader.__version__ import __version__
from civitai_downloader.downloader import civitai_download, url_download, advanced_download, batch_download, version_batch_download
from civitai_downloader.login import login

class CivitaiDownloaderCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="CivitAI Downloader CLI")

        self.parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}", help='Show program version')

        self.subparsers = self.parser.add_subparsers(dest="command", help="Available commands")

        self._add_download_parser()
        self._add_advanced_download_parser()
        self._add_url_download_parser()
        self._add_batch_download_parser()
        self._add_version_batch_download_parser()
        self._add_token_parser()

    def _add_download_parser(self):
        download_parser = self.subparsers.add_parser("download", help="Download models from CivitAI")
        download_parser.add_argument("model_version_id", type=str, help="Model version ID")
        download_parser.add_argument("--local-dir", type=str, default=".", help="Output path")

    def _add_advanced_download_parser(self):
        advanced_download_parser = self.subparsers.add_parser("advanced-download", help="Download models with advanced filtering")
        advanced_download_parser.add_argument("model_version_id", type=str, help="Model version ID")
        advanced_download_parser.add_argument("--local-dir", type=str, default=".", help="Output path")
        advanced_download_parser.add_argument("--type-filter", type=str, help="File type filter")
        advanced_download_parser.add_argument("--format-filter", type=str, help="File format filter")
        advanced_download_parser.add_argument("--size-filter", type=str, help="File size filter")
        advanced_download_parser.add_argument("--fp-filter", type=str, help="File fingerprint filter")

    def _add_url_download_parser(self):
        url_download_parser = self.subparsers.add_parser("url-download", help="Download models from URL")
        url_download_parser.add_argument("model_url", type=str, help="Model URL")
        url_download_parser.add_argument("--local-dir", type=str, default=".", help="Output path")

    def _add_batch_download_parser(self):
        batch_download_parser = self.subparsers.add_parser("batch-download", help="Batch download models")
        batch_download_parser.add_argument("model_id", type=int, help="Model ID")
        batch_download_parser.add_argument("--local-dir", type=str, default=".", help="Output path")

    def _add_version_batch_download_parser(self):
        version_batch_download_parser = self.subparsers.add_parser("version-batch-download", help="Batch download model versions")
        version_batch_download_parser.add_argument("model_version_id", type=int, help="Model version ID")
        version_batch_download_parser.add_argument("--local-dir", type=str, default=".", help="Output path")

    def _add_token_parser(self):
        self.subparsers.add_parser("token", help="Store CivitAI API token")

    def run(self):
        args = self.parser.parse_args()
        if args.command == "download":
            self.download(args.model_version_id, args.local_dir)
        elif args.command == "advanced-download":
            self.advanced_download(args.model_version_id, args.local_dir, args.type_filter, args.format_filter, args.size_filter, args.fp_filter)
        elif args.command == "url-download":
            self.url_download(args.model_url, args.local_dir)
        elif args.command == "batch-download":
            self.batch_download(args.model_id, args.local_dir)
        elif args.command == "version-batch-download":
            self.version_batch_download(args.model_version_id, args.local_dir)
        elif args.command == "login":
            self.store_token()
        else:
            self.parser.print_help()

    def download(self, model_version_id, local_dir):
        civitai_download(model_version_id=model_version_id, local_dir=local_dir)
        print(f"Downloaded model version {model_version_id} to {local_dir}")

    def advanced_download(self, model_version_id, local_dir, type_filter, format_filter, size_filter, fp_filter):
        advanced_download(
            model_version_id=model_version_id,
            local_dir=local_dir,
            type_filter=type_filter,
            format_filter=format_filter,
            size_filter=size_filter,
            fp_filter=fp_filter,
        )
        print(f"Advanced download of model version {model_version_id} to {local_dir}")

    def url_download(self, model_url, local_dir):
        url_download(url=model_url, local_dir=local_dir)
        print(f"Downloaded model from URL {model_url} to {local_dir}")

    def batch_download(self, model_id, local_dir):
        batch_download(model_id=model_id, local_dir=local_dir)
        print(f"Batch downloaded models with ID {model_id} to {local_dir}")

    def version_batch_download(self, model_version_id, local_dir):
        version_batch_download(model_version_id=model_version_id, local_dir=local_dir)
        print(f"Batch downloaded model version {model_version_id} to {local_dir}")

    def store_token(self):
        login()
        print("Login successful")
