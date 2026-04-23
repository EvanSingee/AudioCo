"""Google Drive folder/file download utilities using gdown."""
import os
import re
import logging
import gdown

logger = logging.getLogger(__name__)


def _extract_folder_id(url: str) -> str | None:
    """Pull the Drive folder ID from a share URL."""
    # https://drive.google.com/drive/folders/FOLDER_ID?...
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def download_folder(url: str, dest: str) -> list[str]:
    """
    Download an entire public Google Drive folder into *dest*.
    Returns a sorted list of downloaded file paths.
    """
    os.makedirs(dest, exist_ok=True)
    folder_id = _extract_folder_id(url)
    if folder_id:
        logger.info(f"Downloading Drive folder {folder_id} → {dest}")
        gdown.download_folder(
            id=folder_id,
            output=dest,
            quiet=False,
            use_cookies=False,
        )
    else:
        # Treat as a direct file link
        logger.info(f"Downloading Drive file → {dest}")
        gdown.download(url, output=dest, fuzzy=True, quiet=False)

    # Collect all files
    files = []
    for root, _, fnames in os.walk(dest):
        for f in fnames:
            files.append(os.path.join(root, f))
    return sorted(files)
