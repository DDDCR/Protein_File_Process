import h5py

def count_items(h5_file_path):
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            num_datasets = 0
            num_groups = 0

            def visit(name, obj):
                nonlocal num_datasets, num_groups
                if isinstance(obj, h5py.Dataset):
                    num_datasets += 1
                elif isinstance(obj, h5py.Group):
                    num_groups += 1

            h5_file.visititems(visit)
            # Subtract 1 from groups if you don't want to count the root group
            num_groups -= 1  # Root group '/' is not usually counted

            return num_datasets, num_groups

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    # Replace the path below with your actual file path
    h5_file_path = r"C:\Users\DCR\Documents\University\project\GVAEs\Project\data\raw_unique\adj_matrix.h5"

    datasets, groups = count_items(h5_file_path)
    if datasets is not None:
        print(f"Number of datasets in '{h5_file_path}': {datasets}")
        print(f"Number of groups in '{h5_file_path}': {groups}")

