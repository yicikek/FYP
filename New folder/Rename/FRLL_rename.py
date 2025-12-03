import os

morph_dir = r"C:\Users\kekyi\Downloads\FYP\D-MAD\FRLL_preprocessed\morphed"

pair_counts = {}

for filename in os.listdir(morph_dir):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    # Example: M_173_03-140_03_C07....
    if "M" in parts[0]:
        ids = parts[1]  # "173_03-140_03"
    else:
        ids = parts[0]  # in case already cleaned

    id1, id2 = ids.split("-")

    pair_key = f"{id1}-{id2}"

    if pair_key not in pair_counts:
        pair_counts[pair_key] = 1
    else:
        pair_counts[pair_key] += 1

    new_filename = f"{pair_key}_{pair_counts[pair_key]}.jpg"

    old_path = os.path.join(morph_dir, filename)
    new_path = os.path.join(morph_dir, new_filename)

    os.rename(old_path, new_path)

print("✅ FRLL morph files renamed to match FEI style!")
