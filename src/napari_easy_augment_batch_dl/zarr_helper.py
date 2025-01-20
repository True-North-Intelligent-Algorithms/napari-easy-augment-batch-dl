import zarr
import numpy as np
from numcodecs import JSON
import os


def get_zarr_store(zarr_path):
    
    z = zarr.open(zarr_path, mode='r+')

    return z

def manage_zarr_store(zarr_path, file_names, image_shape=(256, 256), dtype='i4'):
    """
    Manage a Zarr store based on the current file names.
    
    Args:
        zarr_path (str): Path to the Zarr store.
        file_names (list of str): List of file names to manage in the Zarr store.
        image_shape (tuple): Shape of each image.
        
    Returns:
        zarr.Group: The managed Zarr group.
    """
    num_file_names = len(file_names)

    file_names_str = []

    for i in range(num_file_names):
        file_names_str.append(str(file_names[i]))
    
    if not os.path.exists(zarr_path):
        # Case 1: No Zarr store, just starting
        z = zarr.open(zarr_path, mode='w')
        
        # Create datasets
        z.create_dataset(
            'images',
            shape=(num_file_names, *image_shape),
            #chunks=(1, *image_shape),
            dtype=dtype,
            fill_value=0,
            #compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2)
        )
        try:
            z.create_dataset(
                'filenames',
                shape=(num_file_names,),
                dtype=object,
                chunks=(num_file_names,),
                object_codec=JSON(),
            )
        except Exception as e:
            # print exception
            print('Exception occurred when creating filenames dataset ', e)
            
        # Populate filenames
        for i, fname in enumerate(file_names_str):
            z['filenames'][i] = fname
        
        print("Created new Zarr store.")
    
    else:
        # Case 2 or 3: Existing Zarr store
        z = zarr.open(zarr_path, mode='r+')
        
        existing_filenames = list(z['filenames'][:])
        existing_num_files = len(existing_filenames)
        
        if existing_num_files == num_file_names:  #and existing_filenames == file_names_str:
            # Case 2: Same number of files, no changes
            print("Zarr store already up-to-date.")
        
        else:
            # Case 3: New files added
            # Find the new file names
            new_filenames = [fname for fname in file_names_str if fname not in existing_filenames]
            
            # Resize datasets to accommodate new files
            new_num_files = len(existing_filenames) + len(new_filenames)
            z['results'].resize(new_num_files, axis=0)
            z['filenames'].resize(new_num_files)
            
            # Add new file names and placeholders for their results
            for i, fname in enumerate(new_filenames, start=existing_num_files):
                z['filenames'][i] = fname
                z['results'][i] = np.nan  # Placeholder for unprocessed images
            
            print(f"Added {len(new_filenames)} new file names to the Zarr store.")
    
    return z
