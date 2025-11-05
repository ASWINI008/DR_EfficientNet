import zipfile
import os

def load_data():
 
    project_path = r"C:\Users\aswin\Documents\DR_EfficientNet"
    archive_zip = r"C:\Users\aswin\Downloads\archive-20251105T042901Z-1-001.zip"
    extract_path = os.path.join(project_path, "archive", "grayscale_images")

    os.makedirs(extract_path, exist_ok=True)

    if not os.path.exists(archive_zip):
        raise FileNotFoundError(f"ZIP file not found at {archive_zip}")

    print("Starting dataset extraction...")
    try:
        with zipfile.ZipFile(archive_zip, 'r') as zip_ref:
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_path)
                print(f"Extracted: {file}")
        print("\nDataset extracted successfully at:", extract_path)
        return extract_path
    except zipfile.BadZipFile:
        print("Error: The ZIP file is corrupted or incomplete.")
    except Exception as e:
        print("An unexpected error occurred:", e)
