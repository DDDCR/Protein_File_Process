import pandas as pd
import glob

# Define the pattern to match your ligand CSV files
file_list = [r'C:\Users\DCR\Documents\University\project\GVAEs\ligand data\rcsb_pdb_ligand__00001-02500.csv', 
             r'C:\Users\DCR\Documents\University\project\GVAEs\ligand data\rcsb_pdb_ligand_02501-05000.csv', 
             r'C:\Users\DCR\Documents\University\project\GVAEs\ligand data\rcsb_pdb_ligand_05001-07500.csv', 
             r'C:\Users\DCR\Documents\University\project\GVAEs\ligand data\rcsb_pdb_ligand_07501-10000.csv']

# Initialize an empty list to hold DataFrames
df_list = []

for file in file_list:
    # Read each CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Keep only the columns 'Entry ID'
    df = df[['Entry ID']]
    
    # Append the cleaned DataFrame to the list
    df_list.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# Remove duplicate ligands based on 'Ligand ID' and 'Ligand Formula'
combined_df = combined_df.drop_duplicates(subset=['Entry ID'])

# Save the final DataFrame to a CSV file
combined_df.to_csv(r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\Entry_ID.csv', index=False)

