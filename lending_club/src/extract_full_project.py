import zipfile
import glob
import os
import shutil

zip_dir = "e:/Barclays-ForkIT/lending_club_data"
base_dir = "e:/Barclays-ForkIT"

zip_files = sorted(glob.glob(os.path.join(zip_dir, "lending_club*.zip")))

if not zip_files:
    print("No zip files found in", zip_dir)

for f in zip_files:
    print(f"Extracting {os.path.basename(f)}...")
    try:
        with zipfile.ZipFile(f) as z:
            for info in z.infolist():
                if info.filename.startswith("/") or ".." in info.filename:
                    continue
                
                relative_path = os.path.normpath(info.filename)
                
                # To prevent overriding existing flattened CSVs with the ones nested in folders, 
                # we just gracefully extract and only overwrite if it's the same path
                # BUT wait, previously the csv files were extracted to base_dir/lending_club/...
                
                extracted_path = os.path.join(base_dir, relative_path)
                
                if os.path.exists(extracted_path):
                    if not info.is_dir() and os.path.isdir(extracted_path):
                        extracted_path += "_file"
                    elif info.is_dir() and os.path.isfile(extracted_path):
                        extracted_path += "_dir"
                        
                if info.is_dir():
                    os.makedirs(extracted_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                    with z.open(info) as source, open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)
    except Exception as e:
        print(f"Error extracting {f}: {e}")

print("\nDeleting zip files...")
for f in zip_files:
    try:
        os.remove(f)
        print(f"Deleted {os.path.basename(f)}")
    except Exception as e:
        print(f"Failed to delete {f}: {e}")

print("\nDone extracting full project files and cleaning up.")
