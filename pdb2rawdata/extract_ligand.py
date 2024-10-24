'''
put file in the dir of protein pdb files

extract lignad information from protein pdb file
also remove the ion ligand
only save unique ligand
Save protein file with unique ligand extracted to another directory
Save ligand IDs in a text file separated by spaces
'''

import os
import glob
import pymol2

# Define the list of ion residue names to exclude
ion_residues = ['NA','ZN', 'CL' 'MG', 'CA', 'K', 'MN', 'FE', 'CO', 'CU', 'NI', 'CD', 'HG', 'MN', 'SM', 'RB', 'CS', 'BA', 'PT','F', 'BR', 'I']

# Output directory
output_ligand_dir = 'extracted_ligands_unique'
output_protein_dir = 'protein_for_unique_ligand'

if not os.path.exists(output_ligand_dir):
    os.makedirs(output_ligand_dir)

if not os.path.exists(output_protein_dir):
    os.makedirs(output_protein_dir)

# Function to identify potential ions based on residue properties
def is_potential_ion(residue_name, atoms):
    # If residue name is in ion_residues list
    if residue_name.strip().upper() in ion_residues:
        return True
    # If the residue has very few atoms (e.g., ≤ 2)
    if len(atoms) <= 2:
        return True
    return False

# Get all PDB files in the current directory
pdb_files = glob.glob('*.pdb')
ligand_id_list = []

with pymol2.PyMOL() as pymol:
    cmd = pymol.cmd
    for pdb_file in pdb_files:
        print(f'Processing {pdb_file}')
        cmd.reinitialize()
        cmd.load(pdb_file)

        # Remove solvent
        cmd.remove('solvent')

        # Select all heteroatoms (potential ligands)
        cmd.select('potential_ligands', 'hetatm')

        # Get model of potential ligands
        model = cmd.get_model('potential_ligands')

        # Group atoms by residue
        residues = {}
        for atom in model.atom:
            res_id = (atom.chain, atom.resi, atom.resn)
            if res_id not in residues:
                residues[res_id] = []

            residues[res_id].append(atom)

        # Check for the first non-ion ligand and save it
        saved = False
        entry_id = os.path.splitext(os.path.basename(pdb_file))[0]
        for res_id, atoms in residues.items():
            chain, resi, resn = res_id
            if not is_potential_ion(resn, atoms):
                ligand_id = f'{resn}'
                
                output_ligand_file = os.path.join(output_ligand_dir, f'{ligand_id}.pdb')
                output_protein_file = os.path.join(output_protein_dir, f'{entry_id}_with_{ligand_id}.pdb')
                
                if not os.path.exists(output_ligand_file):
                    ligand_selection = f'chain {chain} and resi {resi} and resn {resn}'
                    cmd.select('ligand', ligand_selection)
                    cmd.save(output_ligand_file, 'ligand')
                    cmd.save(output_protein_file, 'all')
                    ligand_id_list.append(ligand_id)

                    print(f'Saved {output_ligand_file} containing ligand {resn} in chain {chain}')
                    print(f'Corresponding protein file saved as {output_protein_file}')
                    break
                else:
                    print(f'File {output_ligand_file} already exists. Skipping duplicate.')

        if not saved:
            print(f'No suitable ligands found in {pdb_file}')

    # Save ligand IDs to a text file
    with open('ligand_ids.txt', 'w') as f:
        f.write(' '.join(ligand_id_list))

    print("all done")
    print("Ligand IDs saved:", ' '.join(ligand_id_list))





