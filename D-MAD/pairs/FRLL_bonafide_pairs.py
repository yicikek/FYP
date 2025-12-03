import os
import itertools

# ------------------------
# Folder and output paths
# ------------------------
bonafide_dir = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL_preprocessed\bonafide"
output_file  = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL_bonafide.txt"

# Only include selected codes (you can adjust if needed)
valid_codes = {"02","03","04","07","09", "08"}

# ------------------------
# List all JPG files
# ------------------------
bonafide_files = sorted([f for f in os.listdir(bonafide_dir) if f.lower().endswith(".jpg")])

print("Total bonafide images:", len(bonafide_files))
print("Sample files:", bonafide_files[:6])

# ------------------------
# Group by person (before '_') and filter by code (after '_')
# ------------------------
grouped = {}
for f in bonafide_files:
    if "_" in f:
        person_id, code = f.split("_")
        code = code.split(".")[0]  # remove .jpg
        if code in valid_codes:
            grouped.setdefault(person_id, []).append(f)

# ------------------------
# Generate all combinations within same person
# ------------------------
pairs = []
for person, files in grouped.items():
    if len(files) >= 2:
        for a, b in itertools.combinations(files, 2):
            pairs.append(f"{a} {b}")

# ------------------------
# Save pairs to file
# ------------------------
with open(output_file, "w") as f:
    f.write("\n".join(pairs))

print(f"✅ Done! Created bonafide_pairs.txt with {len(pairs)} pairs.")
print(f"📄 Saved to: {output_file}")
