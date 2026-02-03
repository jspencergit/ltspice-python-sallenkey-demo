#LTSpice_Cleanup.py
import os

# List of file extensions to delete
extensions_to_delete = {'.raw', '.net', '.log'}

# Iterate through all files in the current directory
for filename in os.listdir('.'):
    file_path = os.path.join('.', filename)
    
    # Check if it's a file (not a directory) and has an extension to delete
    if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in extensions_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

print("Deletion process completed.")