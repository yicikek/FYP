import os

folder = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL_preprocessed\FRLL_bonafide"  # change to your folder path

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        parts = filename.split("_")

        # Need at least 2 parts to form "001_08"
        if len(parts) >= 2:
            new_name = parts[0] + "_" + parts[1] + ".jpg"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            # rename
            os.rename(old_path, new_path)
            print(f"{filename}  -->  {new_name}")
