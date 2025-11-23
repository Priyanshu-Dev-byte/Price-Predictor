import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from requests.adapters import HTTPAdapter, Retry
from urllib.parse import urlparse, unquote


def download_image(image_link, savefolder, timeout=10, max_retries=2):
    if not isinstance(image_link, str) or not image_link:
        return
    try:
        parsed = urlparse(image_link)
        filename = os.path.basename(parsed.path) or 'image'
        filename = unquote(filename.split('?')[0])
        image_save_path = os.path.join(savefolder, filename)
        if os.path.exists(image_save_path):
            return

        session = requests.Session()
        retries = Retry(total=max_retries, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        with session.get(image_link, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(image_save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as ex:
        print(f'Warning: download failed for {image_link}: {ex}')
    return

def download_images(image_links, download_folder, pool_size=None):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder, exist_ok=True)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    cpu = os.cpu_count() or 4
    pool_size = pool_size or min(8, cpu)
    with multiprocessing.Pool(pool_size) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
# ...existing code...
