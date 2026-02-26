import os
import zipfile
from pathlib import Path

def zip_directory(folder_path, output_path):
    folder_path = Path(folder_path)
    # Enable ZIP64 for large files > 4GB or > 65535 files
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Add directory entries (to include empty folders like Input)
            for d in dirs:
                dir_path = Path(root) / d
                arcname = dir_path.relative_to(folder_path)
                zipf.write(dir_path, arcname)
                
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(folder_path)
                print(f"Adding: {arcname}")
                zipf.write(file_path, arcname)
    print(f"Delivery zip archive created at: {output_path}")

if __name__ == "__main__":
    source = "i:/freelance/SCA-Smartcard-Pipeline-3/SCA-Smartcard-ML-Pipeline"
    dest = "i:/freelance/SCA-Smartcard-Pipeline-3/SCA-Smartcard-ML-Pipeline_Delivery.zip"
    zip_directory(source, dest)
