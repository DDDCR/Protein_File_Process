import os
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from openbabel import openbabel as ob
from openbabel import pybel

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
        self.atom_data = []
        self.graph = None
        self.node_features = None
        self.sybyl_types = None
        self.node_features_normalized = None
        self.scaler = None  # To store the scaler for inverse transformations if needed
        self.adj_matrix = None
        
        self.sybyl_map = {'Al': 0, 'Br': 1, 'C.1': 2, 'C.2': 3, 'C.3': 4, 'C.ar': 5, 'C.cat': 6, 'Ca': 7, 'Cl': 8, 'Du': 9, 'F': 10, 
             'H': 11, 'H.spc': 12, 'H.t3p': 13, 'I': 14, 'K': 15, 'Li': 16, 'LP': 17, 'N.1': 18, 'N.2': 19, 'N.3': 20, 
             'N.4': 21, 'N.am': 22, 'N.ar': 23, 'N.pl3': 24, 'Na': 25, 'O.2': 26, 'O.3': 27, 'O.co2': 28, 'O.spc': 29, 'O.t3p': 30, 
             'P.3': 31, 'S.2': 32, 'S.3': 33, 'S.o': 34, 'S.o2': 35, 'Si': 36} # To map the sybyl atom types to integer index


    def read_pdb(self):
        """
        Manually read the PDB file to extract atom indices, types, and coordinates.
        """
        
        with open(self.pdb_file, 'r') as file:
            for line in file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_type = line[76:78].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if not self.remove_hs or (self.remove_hs and atom_type != 'H'):
                        self.atom_data.append((atom_type, x, y, z))
        return self.atom_data

    def create_graph_and_distanceM(self):
        """
        Create graph from atom data and calculate the distance matrix.
        """
        self.graph = nx.Graph()
        coords = []
        for idx, (atom_type, x, y, z) in enumerate(self.atom_data):
            self.graph.add_node(idx, symbol=atom_type, x=x, y=y, z=z)
            coords.append((x, y, z))


        coords = np.array(coords)
        distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
        self.distance_matrix = pd.DataFrame(distance_matrix)


    def translate_sybyl_types(self):  # from loic code
        """
        Translate element types to sybyl type. Reads a PDB file using Openbabel package and
        returns a list of translated element types.
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
        num_nodes = self.graph.number_of_nodes()
        
        # Initialize node features array: [atomic_number, x, y, z]
        node_features_basic = np.zeros((num_nodes, 4))
        
        for idx, data in self.graph.nodes(data=True):
            atom_symbol = data['symbol']
            atomic_num = Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), atom_symbol)
            x = data['x']
            y = data['y']
            z = data['z']
            node_features_basic[idx] = [atomic_num, x, y, z]
        
        # Map SYBYL types to integer indices
        self.map_sybyl_types()
        sybyl_indices_array = np.array(self.sybyl_indices).reshape(-1, 1)  # Shape: (num_nodes, 1)

        
        # Concatenate basic features with Sybyl encoded features
        self.node_features = np.hstack((node_features_basic, sybyl_indices_array))  # Shape: (num_nodes, 4 + num_sybyl_features)
        print(f"Node features shape: {self.node_features.shape}")
        

    def normalize_data(self):
        """
        Normalize the node features for VAE encoding.
        """
        self.scaler = StandardScaler()
        self.node_features_normalized = self.scaler.fit_transform(self.node_features)




    def preprocess(self, normalize=True):
        """
        Execute all preprocessing steps.
        """
        self.read_pdb()
        self.create_graph_and_distanceM()
        self.translate_sybyl_types()
        self.prepare_data()
        
        if normalize:
            self.normalize_features()
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
    
    
    def preprocess_and_save(self, output_dir, normalize=True):
        """
        Execute preprocessing and save the node features and distance matrix into a single CSV file.

        Parameters:
        - output_dir (str): Directory where the output CSV will be saved.
        - normalize (bool): Whether to normalize the node features before saving.
        """
        
        self.preprocess(normalize=normalize)

        output_filename = os.path.basename(self.pdb_file).replace('.pdb', '_processed.csv')
        output_path = os.path.join(output_dir, output_filename)
        
        # Prepare node features DataFrame
        num_sybyl_features = 1  # Since we're using integer indices
        node_feature_labels = ['atomic_number', 'x', 'y', 'z', 'sybyl_type']
        node_features_df = pd.DataFrame(self.node_features_normalized, columns=node_feature_labels)
        

        # Prepare distance matrix DataFrame
        distance_df = self.distance_matrix.copy()
        
        

        # Save to CSV with clear separation
        with open(output_path, 'w') as f:
            # Write Node Features
            f.write("NODE_FEATURES\n")
            node_features_df.to_csv(f)
            f.write("\n")

            # Write Distance Matrix
            f.write("DISTANCE_MATRIX\n")
            distance_df.to_csv(f)
            f.write("\n")
        
        print(f"Data saved to {output_path}")


def process_directory(input_dir, output_dir, normalize=True):
    """
    Process all PDB files in the input directory and save the processed data.

    Parameters:
    - input_dir (str): Directory containing PDB files.
    - output_dir (str): Directory where processed CSV files will be saved.
    - normalize (bool): Whether to normalize node features before saving.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]

    if not pdb_files:
        print(f"No PDB files found in {input_dir}.")
        return

    for filename in pdb_files:
        pdb_path = os.path.join(input_dir, filename)
        try:
            preprocessor = MoleculePreprocessor(pdb_path)
            preprocessor.preprocess_and_save(output_dir, normalize=normalize)
            print(f"Processed and saved data for {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    input_dir = r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\pdb_example'
    output_dir = r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\processed_ligands'
    os.makedirs(output_dir, exist_ok=True)
    process_directory(input_dir, output_dir,normalize=False)