import os
import requests
import pandas as pd
import time

# Read the CSV file containing entry IDs
df = pd.read_csv(r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\Entry_ID.csv')

# Ensure the 'Entry ID' column exists
if 'Entry ID' not in df.columns:
    raise ValueError("The 'Entry ID' column is missing in the CSV file.")

# Remove rows where 'Entry ID' is NaN or empty
df = df[df['Entry ID'].notna() & df['Entry ID'].astype(str).str.strip().astype(bool)]

# Get the list of unique entry IDs
entry_ids = df['Entry ID'].unique()

# Create a directory to store the PDB files
output_dir = 'pdb_files'
os.makedirs(output_dir, exist_ok=True)

# RCSB PDB RESTful API URL template for PDB files
url_template = 'https://files.rcsb.org/download/{}.pdb'


# Loop over each entry ID and download the PDB file
def download_pdb(entry_id):
        url = url_template.format(entry_id.upper())
        response = requests.get(url)
        if response.status_code == 200:
            # Save the PDB file
            file_path = os.path.join(output_dir, f'{entry_id}.pdb')
            with open(file_path, 'w') as file:
                file.write(response.text)
            print(f'Downloaded: {entry_id}')
        else:
            print(f'Failed to download {entry_id}: HTTP {response.status_code}')

start_downloading = False
start_entry_id = '2GT9'  # Replace with the Entry ID you want to start from


for entry_id in entry_ids:
    
    # Check if entry_id is NaN or not a string
    if not isinstance(entry_id, str):
        entry_id = str(entry_id)
    entry_id = entry_id.strip()
    if not entry_id:
        continue  # Skip empty strings

    # Skip Entry IDs until we reach the starting point
    if not start_downloading:
        if entry_id.upper() == start_entry_id.upper():
            start_downloading = True
        else:
            continue  # Skip this Entry ID

    # Check if the file already exists to avoid re-downloading
    file_path = os.path.join(output_dir, f'{entry_id}.pdb')
    if os.path.exists(file_path):
        print(f'File already exists: {entry_id}')
        continue
    
    # Add a delay between requests to avoid overloading the server
    time.sleep(0.2)  # Sleep for 200 milliseconds

    download_pdb(entry_id)

print('All downloads completed.')
