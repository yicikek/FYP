import os

morph_dir = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FEI_preprocessed\morphed"

# To track index for each identity pair
pair_counts = {}

for filename in os.listdir(morph_dir):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    # Expected: M_99-11_96-11_C08_B50_...
    if len(parts) < 3:
        continue

    id1 = parts[1]
    id2 = parts[2]

    pair_key = f"{id1}_{id2}"

    # Increase index count for same identity pair
    if pair_key not in pair_counts:
        pair_counts[pair_key] = 1
    else:
        pair_counts[pair_key] += 1

    # Example: 99-11_96-11_1.jpg
    new_filename = f"{pair_key}_{pair_counts[pair_key]}.jpg"

    old_path = os.path.join(morph_dir, filename)
    new_path = os.path.join(morph_dir, new_filename)

    os.rename(old_path, new_path)

print("✅ Morph files renamed WITHOUT creating new files")
