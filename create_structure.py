import os

# Define the folder structure
folders = [
    ".streamlit",
    ".vscode",
    ".github/workflows",
    "notebooks",
    "tests",
    "app",
    "scripts"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Define files to create with optional content
files = {
    ".streamlit/config.toml": "",
    ".vscode/settings.json": "",
    ".github/workflows/unittests.yml": "",
    ".gitignore": "",
    "requirements.txt": "",
    "README.md": "",
    "notebooks/__init__.py": "",
    "notebooks/example.ipynb": "",
    "notebooks/README.md": "",
    "tests/__init__.py": "",
    "app/__init__.py": "",
    "app/main.py": "# main Streamlit application script",
    "app/utils.py": "# utility functions for data processing and visualization",
    "scripts/__init__.py": "",
    "scripts/README.md": ""
}

# Create files with content
for file, content in files.items():
    with open(file, 'w') as f:
        f.write(content)

print("Folder structure created successfully!")
