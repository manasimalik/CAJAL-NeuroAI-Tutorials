import os
from pathlib import Path
from IPython.display import HTML, display


import requests
import os
from pathlib import Path
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

def download_with_requests(url, filename, destination_dir=".", force_download=False):
    """
    Download using requests library with SSL verification disabled.
    """
    download_url = f"{url}/raw/master/{filename}"
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(destination_dir, filename)
    
    # Check if file already exists
    if os.path.exists(file_path) and not force_download:
        file_size = os.path.getsize(file_path)
        print(f"File already exists: {file_path}")
        print(f"File size: {file_size} bytes")
        print("Skipping download. Use force_download=True to re-download.")
        return file_path
    
    try:
        if os.path.exists(file_path) and force_download:
            print(f"File exists but force_download=True. Re-downloading {filename}...")
        else:
            print(f"Downloading {filename}...")
        
        # Download with SSL verification disabled
        response = requests.get(download_url, stream=True, verify=False)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file:
            downloaded_size = 0
            chunk_size = 8192
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\nDownload completed: {file_path}")
        final_size = os.path.getsize(file_path)
        print(f"File size: {final_size} bytes")
        
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return None



def set_background(color):    
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)
    
    display(HTML('<img src onerror="{}" style="display:none">'.format(script)))