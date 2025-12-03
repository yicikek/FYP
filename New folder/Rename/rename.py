import os
import pandas as pd
import re
import shutil

def rename_morph_files(base_path, dataset_name):
    morph_folder = os.path.join(base_path, "morphed")
    renamed_folder = os.path.join(base_path, "morph_renamed")
    os.makedirs(renamed_folder, exist_ok=True)
    
    for file in os.listdir(morph_folder):
        src_path = os.path.join(morph_folder, file)
        if not os.path.isfile(src_path):
            continue
        
        # 1️⃣ FRLL pattern: e.g., 173_03_117_03_alpha0.5_combined_morph_q48
        if dataset_name == "FRLL":
            match = re.match(r"(\d{3}_\d{2})_(\d{3}_\d{2})", file)
            if match:
                id1, id2 = match.groups()
                new_name = f"{id1}-{id2}.jpg"
                dst_path = os.path.join(renamed_folder, new_name)
                shutil.copy(src_path, dst_path)
                print(f"FRLL renamed: {file} → {new_name}")
        
        # 2️⃣ FEI pattern: e.g., M_99-11_96-11_C08_B50_W50_PA08_PM00_F00
        elif dataset_name == "FEI":
            match = re.search(r"(\d{2}-\d{2})_(\d{2}-\d{2})", file)
            if match:
                id1, id2 = match.groups()
                new_name = f"{id1}_{id2}.jpg"
                dst_path = os.path.join(renamed_folder, new_name)
                shutil.copy(src_path, dst_path)
                print(f"FEI renamed: {file} → {new_name}")

rename_morph_files(BASE_FRLL, "FRLL")
rename_morph_files(BASE_FEI, "FEI")
print("✅ All morph files renamed successfully!")