import os

def add_large_images_to_gitignore(directory, size_limit_mb=10):
    size_limit_bytes = size_limit_mb * 1024 * 1024  # Convert MB to bytes
    gitignore_entries = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.tif')):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > size_limit_bytes:
                    relative_path = os.path.relpath(file_path, directory)
                    gitignore_entries.append(relative_path)

    with open(os.path.join(directory, '.gitignore'), 'a') as f:
        for entry in gitignore_entries:
            f.write(f"{entry}\n")

# Replace '/home/steve/Projects/WeaverLab/STIFMaps' with the path to your local folder
add_large_images_to_gitignore('/home/steve/Projects/WeaverLab/STIFMaps')
