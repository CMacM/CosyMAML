import h5py
import numpy as np

for i in range(1,5):
    # Load the original npz file
    filename = 'data/1000tasks_5000samples_{}4seed.npz'.format(i)
    print('Processing...', filename)
    npz_data = np.load(filename, allow_pickle=True)
    print('NPZ file loaded, creating h5 file...')
    # Save to HDF5
    h5_filename = filename.replace('.npz', '.h5')
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('X_train', data=npz_data['X_train'])
        h5f.create_dataset('y_train', data=npz_data['y_train'])
        h5f.create_dataset('dndz', data=npz_data['dndz'])
        h5f.create_dataset('z', data=npz_data['z'])
        h5f.create_dataset('ells', data=npz_data['ells'])

        # convert dndz_params to an array
        dndz_params = []
        for entry in npz_data['dndz_params']:
            for key, value in entry.items():
                if key == 'z0':
                    z0 = value
                elif key == 'sigma':
                    p2 = value
                    flag = 0
                elif key == 'alpha':
                    p2 = value
                    flag = 1
            dndz_params.append([z0, p2, flag])
        h5f.create_dataset('dndz_params', data=dndz_params)
    print('H5 file created:', h5_filename)
    print('Proceeding to next file...')
print('All files processed.')