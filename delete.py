import json

# --- Load your JSON ---
with open("/source/junhyuk/motion-style/dataset/100style/100style_clean_clean.json", "r") as f:
    data = json.load(f)

# --- Sort keys ---
sorted_keys = sorted(data.keys())

# --- Write index file ---
with open("key_indices.txt", "w") as f:
    for idx, key in enumerate(sorted_keys):
        f.write(f"{idx}: {key}\n")

print("Done. Check 'key_indices.txt'.")