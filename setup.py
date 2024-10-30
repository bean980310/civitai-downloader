import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="civitai-downloader",
    version="0.3.0",
    author="Hyunbeen Chang",
    author_email="bean980310@gmail.com",
    description="A package to download models from CivitAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bean980310/civitai-downloader",
    project_urls={
        "Bug Tracker": "https://github.com/bean980310/civitai-downloader/issues",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'civitai-downloader=civitai_downloader.cli:main',
        ],
    },
    install_requires=[
        'tqdm',
        'urllib3'
    ]
)
