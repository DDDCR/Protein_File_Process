import os
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from openbabel import openbabel as ob
from openbabel import pybel
from scipy.spatial.distance import pdist, squareform
import h5py

class MoleculePreprocessor:
    def __init__(self, pdb_file, remove_hs=True):
        """
        Initialize the MoleculePreprocessor with a PDB file.

        Parameters:
        - pdb_file (str): Path to the PDB file.
        - remove_hs (bool): Whether to remove hydrogen atoms
        """
        self.pdb_file = pdb_file
        self.remove_hs = remove_hs
        self.mol = None
        self.graph = None
        self.node_features = None
        self.sybyl_types = None
        self.adj_matrix = None
        self.node_features_normalized = None
        self.scaler = None  # To store the scaler for inverse transformations if needed
        self.sybyl_map = {'Al': 0, 'Br': 1, 'C.1': 2, 'C.2': 3, 'C.3': 4, 'C.ar': 5, 'C.cat': 6, 'Ca': 7, 'Cl': 8, 'Du': 9, 'F': 10, 
             'H': 11, 'H.spc': 12, 'H.t3p': 13, 'I': 14, 'K': 15, 'Li': 16, 'LP': 17, 'N.1': 18, 'N.2': 19, 'N.3': 20, 
             'N.4': 21, 'N.am': 22, 'N.ar': 23, 'N.pl3': 24, 'Na': 25, 'O.2': 26, 'O.3': 27, 'O.co2': 28, 'O.spc': 29, 'O.t3p': 30, 
             'P.3': 31, 'S.2': 32, 'S.3': 33, 'S.o': 34, 'S.o2': 35, 'Si': 36} # To map the sybyl atom types to integer index


    def read_pdb(self):
        """
        Read the PDB file and load the molecule.
        """
        # Load the molecule from the PDB file
        self.mol = Chem.MolFromPDBFile(self.pdb_file, removeHs=self.remove_hs)
        
        # Check if the molecule was loaded successfully
        if self.mol is None:
            raise ValueError("Failed to load molecule from PDB file.")
        
        # Ensure the molecule has 3D coordinates
        if not self.mol.GetNumConformers():
            # Generate 3D coordinates if not present
            AllChem.EmbedMolecule(self.mol)
            if not self.mol.GetNumConformers():
                raise ValueError("Failed to generate 3D coordinates for the molecule.")

    def create_graph(self):
        """
        Create a graph representation of the molecule, including 3D coordinates.
        """
        self.graph = nx.Graph()
        
        # Get the conformer to access 3D coordinates
        conformer = self.mol.GetConformer()
        
        # Add atoms as nodes with 3D coordinates
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            
            # Get the 3D coordinates of the atom
            pos = conformer.GetAtomPosition(atom_idx)
            x, y, z = pos.x, pos.y, pos.z
            
            # Add node with attributes
            self.graph.add_node(atom_idx, symbol=atom_symbol, x=x, y=y, z=z)
        
        # Add bonds as edges
        for bond in self.mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            self.graph.add_edge(begin_idx, end_idx, bond_type=bond_type)

    def translate_sybyl_types(self):
        """
        Translate element types to sybyl type. Reads a PDB file using Openbabel package and
        returns a list of translated element types.
        Original code from loic.
        """
        obConversion = ob.OBConversion()
        obConversion.SetInFormat('pdb')
        obConversion.SetInAndOutFormats("pdb", "mol2")
        ttab = ob.ttab
        ttab.SetFromType("INT")
        ttab.SetToType("SYB")
        ob.obErrorLog.SetOutputLevel(0)

        l_syb = []  # Empty list for the Sybyl atom types

        # Setting up Openbabel atom type conversion and iterating through the atoms
        mol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.ReadFile(mol, self.pdb_file)
        for obatom in ob.OBMolAtomIter(mol):
            if obatom.GetResidue().GetName() == 'HOH':  # Use a special type for oxygens in water molecules
                l_syb.append('O.3.wat')
            else:
                l_syb.append(str(ttab.Translate(obatom.GetType())).capitalize())

        self.sybyl_types = l_syb  # Saves the list of translated sybyl types



    def map_sybyl_types(self):
        """
        Map SYBYL atom types to their corresponding integer indices using the sybyl_map.
        """

        if not self.sybyl_types:
            raise ValueError("SYBYL types have not been translated. Call translate_sybyl_types() first.")

        self.sybyl_indices = []
        for sybyl in self.sybyl_types:
            if sybyl in self.sybyl_map:
                self.sybyl_indices.append(self.sybyl_map[sybyl])

    def prepare_data(self):
        """
        Prepare node features and adjacency matrix for VAE encoding.
        """
        h_filtered_graph = nx.subgraph(self.graph, [n for n, attr in self.graph.nodes(data=True) if attr['symbol'] != 'H'])

        num_nodes = h_filtered_graph.number_of_nodes()
        
        # Initialize node features array: [atomic_number, x, y, z]
        node_features_basic = np.zeros((num_nodes, 4))
        
        positions = []  # List to store coordinates for distance matrix calculation
        
        for idx, data in self.graph.nodes(data=True):
            atom_symbol = data['symbol']
            atomic_num = Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), atom_symbol)
            x, y, z = data['x'], data['y'], data['z']
            positions.append([x, y, z])
            node_features_basic[idx] = [atomic_num, x, y, z]
        
        # Convert list of positions to numpy array for distance matrix calculation
        positions_array = np.array(positions)
        
        # Calculate the Euclidean distance matrix
        self.distance_matrix = squareform(pdist(positions_array))
        
        # Map SYBYL types to integer indices
        self.map_sybyl_types()
        sybyl_indices_array = np.array([self.sybyl_indices[n] for n in h_filtered_graph]).reshape(-1, 1)  # Shape: (num_nodes, 1)

        print("node_features_basic shape:", node_features_basic.shape)
        print("sybyl_indices_array shape:", sybyl_indices_array.shape)

        # Concatenate basic features with Sybyl encoded features
        self.node_features = np.hstack((node_features_basic, sybyl_indices_array))  # Shape: (num_nodes, 4 + num_sybyl_features)
        
        # Get adjacency matrix
        self.adj_matrix = nx.adjacency_matrix(self.graph).todense()

    def normalize_data(self):
        """
        Normalize the node features for VAE encoding.
        """
        self.scaler = StandardScaler()
        self.node_features_normalized = self.scaler.fit_transform(self.node_features)

    def preprocess(self, normalize=False):
        """
        Execute all preprocessing steps.
        """
        self.read_pdb()
        self.create_graph()
        self.translate_sybyl_types()
        self.prepare_data()
        if normalize:
            self.normalize_data()
        else:
            self.node_features_normalized = self.node_features

    def visualize_graph(self):
        """
        Visualize the molecule graph in 3D (optional).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

        # Extract node positions
        node_positions = {idx: (data['x'], data['y'], data['z']) for idx, data in self.graph.nodes(data=True)}
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw nodes
        xs, ys, zs = zip(*node_positions.values())
        ax.scatter(xs, ys, zs, s=100, c='blue', depthshade=True)
        
        # Draw edges
        for u, v in self.graph.edges():
            x_coords = [node_positions[u][0], node_positions[v][0]]
            y_coords = [node_positions[u][1], node_positions[v][1]]
            z_coords = [node_positions[u][2], node_positions[v][2]]
            ax.plot(x_coords, y_coords, z_coords, c='black')
        
        # Annotate atoms with sybyl atom types
        for idx, (x, y, z) in node_positions.items():
            sybyl_type = self.sybyl_types[idx] if self.sybyl_types and idx < len(self.sybyl_types) else self.graph.nodes[idx]['symbol']
            ax.text(x, y, z, sybyl_type, size=10, zorder=1, color='k')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()



def process_pdb_files(input_dir, output_dir, normalize=False):
    """
    Process all PDB files in the input directory and save node features,
    adjacency matrices, and distance matrices into separate HDF5 files.

    Parameters:
    - input_dir (str): Path to the directory containing PDB files.
    - output_dir (str): Path to the directory where HDF5 files will be saved.
    - normalize (bool): Whether to normalize node features.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for the HDF5 files
    node_features_path = os.path.join(output_dir, 'node_features.h5')
    adj_matrix_path = os.path.join(output_dir, 'adj_matrix.h5')
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix.h5')

    # Open HDF5 files in write mode
    with h5py.File(node_features_path, 'w') as hf_node, \
         h5py.File(adj_matrix_path, 'w') as hf_adj, \
         h5py.File(distance_matrix_path, 'w') as hf_dist:
        
        # Iterate over all PDB files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.pdb'):
                pdb_path = os.path.join(input_dir, filename)
                
                try:
                    # Initialize preprocessor
                    preprocessor = MoleculePreprocessor(pdb_path, remove_hs=True)
                    
                    # Preprocess the molecule
                    preprocessor.preprocess(normalize=normalize)
                    
                    # Extract data
                    node_features = preprocessor.node_features_normalized
                    adj_matrix = np.array(preprocessor.adj_matrix)
                    distance_matrix = preprocessor.distance_matrix
                    
                    # Define a unique dataset name, e.g., based on filename without extension
                    dataset_name = os.path.splitext(filename)[0]
                    
                    # Save to HDF5 files
                    hf_node.create_dataset(dataset_name, data=node_features, compression="gzip")
                    hf_adj.create_dataset(dataset_name, data=adj_matrix, compression="gzip")
                    hf_dist.create_dataset(dataset_name, data=distance_matrix, compression="gzip")
                    
                    print(f"Saved data for {filename}.")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print("Processing completed. HDF5 files saved.")

if __name__ == "__main__":
    # Define the input and output directories
    input_directory = r"C:\Users\DCR\Documents\University\project\Protein File Process\data\AHG"
    output_directory = r"C:\Users\DCR\Documents\University\project\Link Prediction\data\AHG_raw"

    # Call the processing function
    process_pdb_files(input_directory, output_directory, normalize=False)