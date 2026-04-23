import h5py
import numpy as np
import glob
import sys
import os
from natsort import natsorted

data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

for pattern, output in [("train*", "all_train"), ("valid*", "all_valid"), ("test*", "all_test")]:
    files = natsorted(glob.glob(f"{data_dir}/{pattern}.h5"))
    if not files:
        continue
    
    data_list, target_list, coords_list = [], [],[]
    for f in files:
        with h5py.File(f, 'r') as h5f:
            data_list.append(h5f["sequence"][:])
            target_list.append(h5f["target"][:])
            coords_list.append(h5f["coords"][:])
    
    with h5py.File(f"{data_dir}/{output}.h5", "w") as h5f_out:
        h5f_out.create_dataset("sequence", data=np.concatenate(data_list))
        h5f_out.create_dataset("target", data=np.concatenate(target_list))
        h5f_out.create_dataset("coords", data=np.concatenate(coords_list))
        h5f_out.close()
    
    print(f"Created {output}.h5 from {len(files)} files")
    
    # # Remove original files
    for f in files:
        os.remove(f)
        #print(f"  Removed {os.path.basename(f)}")
    




# Read the three files in order: train, val, test
seq_list, tgt_list, coord_list = [], [], []

for name in ["all_train.h5", "all_valid.h5", "all_test.h5"]:
    filepath = os.path.join(data_dir, name)

    try:
        with h5py.File(filepath, "r") as h5f:
            seq_list.append(h5f["sequence"][:])
            tgt_list.append(h5f["target"][:])
            coord_list.append(h5f["coords"][:])   

        print(f"Read {filepath}")

    except FileNotFoundError:
        print(f"Warning: {filepath} not found, skipping...")
    except KeyError as e:
        print(f"Warning: dataset {e} missing in {filepath}, skipping...")

# Write combined dataset
if seq_list:
    out_path = os.path.join(data_dir, "train_data.h5")

    with h5py.File(out_path, "w") as h5f_out:
        h5f_out.create_dataset("sequence", data=np.concatenate(seq_list))
        h5f_out.create_dataset("target", data=np.concatenate(tgt_list))
        h5f_out.create_dataset("coords", data=np.concatenate(coord_list))

    total_samples = sum(arr.shape[0] for arr in seq_list)

    print(f"\nCreated {out_path}")
    print(f"Total samples: {total_samples}")
else:
    print("\nNo files were successfully read!")

# remove the redundant files
for name in ["all_train.h5", "all_valid.h5", "all_test.h5"]:
    path = os.path.join(data_dir, name)
    print(path)
    if os.path.exists(path):
        os.remove(path)