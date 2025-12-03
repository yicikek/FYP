import os

# --- PATHS ---
pairs_file = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI\bonafide_list.txt"
failed_folder = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI_failed\bonafide"
output_file = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI\pairs_cleaned.txt"

# --- Collect failed image base names (without extension) ---
failed_names = {os.path.splitext(f)[0] for f in os.listdir(failed_folder)}
print(f"Found {len(failed_names)} failed images.")

# --- Read and clean pairs ---
cleaned_lines = []
with open(pairs_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        img1, img2 = parts
        name1 = os.path.splitext(os.path.basename(img1))[0]
        name2 = os.path.splitext(os.path.basename(img2))[0]

        # Skip this pair if either image is failed
        if name1 in failed_names or name2 in failed_names:
            continue
        cleaned_lines.append(line)

# --- Save the cleaned pairs ---
with open(output_file, "w") as f:
    f.writelines(cleaned_lines)

print(f"✅ Done. Cleaned pairs saved to: {output_file}")
print(f"Original pairs: {len(open(pairs_file).readlines())}")
print(f"Cleaned pairs : {len(cleaned_lines)}")
