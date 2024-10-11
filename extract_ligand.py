'''
put file in the dir of protein pdb files

extract lignad information from protein pdb file
also remove the ion ligand
only save unique ligand
'''

import os
import glob
import pymol2

# Define the list of ion residue names to exclude
ion_residues = ['NA','ZN', 'CL' 'MG', 'CA', 'K', 'MN', 'FE', 'CO', 'CU', 'NI', 'CD', 'HG', 'MN', 'SM', 'RB', 'CS', 'BA', 'PT','F', 'BR', 'I']

# Output directory
output_dir = 'extracted_ligands_unique'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        for res_id, atoms in residues.items():
            chain, resi, resn = res_id
            if not is_potential_ion(resn, atoms):
                ligand_id = f'{resn}'
                output_file = os.path.join(output_dir, f'{ligand_id}.pdb')
                
                if not os.path.exists(output_file):
                    ligand_selection = f'chain {chain} and resi {resi} and resn {resn}'
                    cmd.select('ligand', ligand_selection)
                    cmd.save(output_file, 'ligand')
                    print(f'Saved {output_file} containing ligand {resn} in chain {chain}')
                    break
                else:
                    print(f'File {output_file} already exists. Skipping duplicate.')

        if not saved:
            print(f'No suitable ligands found in {pdb_file}')

    print("all done")





