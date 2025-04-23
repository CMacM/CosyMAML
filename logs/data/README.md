Job 2869 - Small scale test of MAML pipeline
nodes = 1
cpus_per_node = 32
batch_size = 5
n_samples = 500
n_tasks = 20
force_stop = 5
n_check = 100
max_iter = 1000

Job 2870 - Replication of MAML paper (10s invtervals)
nodes = 1
cpus_per_node = 32
batch_size = 5
n_samples = 500
n_tasks = 20
force_stop = 500
n_check = 1000
max_iter = 100000

Job 2897 - Upscaled test of MAML pipeline (1s intervals)
nodes = 1
cpus_per_node = 32
batch_size = 10
n_tasks = 80
n_samples = 1000
force_stop = 500
n_check = 1000
max_iter = 1000000

Job 2950 - Replication of MAML paper w/ npz (1s intervals)
nodes = 1
cpus_per_node = 32
batch_size = 5
n_samples = 500
n_tasks = 20
force_stop = 500
n_check = 1000
max_iter = 100000

Job 2976 - Replication of MAML paper w/ hdf5 (1s intervals)
nodes = 1
cpus_per_node = 32
batch_size = 5
n_samples = 500
n_tasks = 20
force_stop = 500
n_check = 1000
max_iter = 100000