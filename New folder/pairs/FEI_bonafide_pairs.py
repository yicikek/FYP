import os
import itertools

# Folder containing all images like 1-01.jpg, 1-02.jpg, etc.
bonafide_dir = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI_preprocessed\bonafide"

# Output file
bonafide_pairs_file = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI_bonafide.txt"

# Only use these emotion/angle codes
valid_codes = {"04", "05", "06", "07", "11", "12", "13"}

# List only JPG files
bonafide_files = sorted([f for f in os.listdir(bonafide_dir) if f.lower().endswith(".jpg")])

# Group images by person number (before the "-")
grouped = {}
for f in bonafide_files:
    if "-" in f:
        person_id, code = f.split("-")
        code = code.split(".")[0]  # remove .jpg
        if code in valid_codes:
            grouped.setdefault(person_id, []).append(f)

# Create bonafide-bonafide pairs (within same person)
bonafide_pairs = []
for person, files in grouped.items():
    if len(files) >= 2:
        # make all unique pairs (e.g., 03-04, 03-05, etc.)
        for a, b in itertools.combinations(files, 2):
            bonafide_pairs.append(f"{a} {b}")

# Save to file
with open(bonafide_pairs_file, "w") as f:
    f.write("\n".join(bonafide_pairs))

print(f"✅ Done! Created bonafide_pairs.txt with {len(bonafide_pairs)} pairs.")
print(f"📄 Saved to: {bonafide_pairs_file}")
