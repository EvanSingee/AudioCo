import os
import requests
import zipfile
import shutil
import streamlit as st

def download_and_extract_drive_folder(drive_url: str, dest_dir: str) -> str:
    """
    Download and extract a Google Drive folder shared as a zip link.
    Assumes the link is a Google Drive folder, and user has zipped it manually.
    """
    # Try to convert folder link to download link (user must provide a zip)
    if "drive.google.com" in drive_url and "folders" in drive_url:
        st.error("Google Drive folders must be zipped and shared as a direct download link.")
        return ""
    if not drive_url.endswith(".zip"):
        st.error("Please provide a direct link to a .zip file.")
        return ""
    os.makedirs(dest_dir, exist_ok=True)
    local_zip = os.path.join(dest_dir, "dataset.zip")
    with requests.get(drive_url, stream=True) as r:
        r.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(local_zip)
    # Move audio files to dest_dir root if needed
    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
                src = os.path.join(root, f)
                dst = os.path.join(dest_dir, f)
                if src != dst:
                    shutil.move(src, dst)
    return dest_dir
