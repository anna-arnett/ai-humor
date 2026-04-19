import json

with open("humor_examples.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples.")

required_keys = {
    "id",
    "source",
    "C_text",
    "X_text",
    "X_speaker",
    "Y",
    "gold_support_label",
    "gold_flags",
    "gold_hallucination_flag",
    "gold_note",
}

for i, row in enumerate(data):
    missing = required_keys - set(row.keys())
    if missing:
        print(f"Example index {i} is missing keys: {missing}")

print("Done.")