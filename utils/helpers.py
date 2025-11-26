import os
import json

# Load a JSON file and return its content as a Python dictionary
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Save a Python dictionary as a JSON file at the specified path
def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Make sure the output folder exists
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)    # Save with readable formatting

# List all .json files in a folder
def list_json_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(".json")]