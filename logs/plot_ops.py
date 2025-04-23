#!/usr/bin/env python3
import matplotlib.pyplot as plt
import re
import argparse
"""
This script processes and visualizes job performance data from log files. These logs files come from the glljobstat tool."
"""

# courbe tendance iops, pas seulement volumétrie

# courbe évolution chronologique nombre iops par leur taille

# test ior en fonction des tailles d'opérations pour tester les performances (ior 16M, lecture 4K Random, 2k, 32k, lecture random (-z sur ior), -i 3 (ou 5), 
# taille des IO fait par -t (4k,32k...), -b option volume de données par thread (volume ~32G ou 64 pour 1 thread), le fichier doit faire au minimum 64G, avec -n 1 pour 1 thread, 2 4 8..)
# pour >1 thread sur IOR passer par mpi et ior réglages aussi

def get_color(size,size_to_color):
    """Returns a color for a given size"""
    return size_to_color.get(size, "black")

# Input values can be in bytes, KB, MB, GB
# Output value is in GB
def convert_size(size_str): # Function to convert size to GB
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    for unit, factor in units.items():
        if size_str.endswith(unit):
            return (int(size_str.strip(unit)) * factor) / (1024**3)
    return int(size_str) / (1024**3) 

def filter_zero_intervals_with_none(y, timestamps): # Function to filter zero intervals, to make plot more readable
    if not y or not timestamps or len(y) != len(timestamps):
        raise ValueError("Lists must have the same length and must not be empty.")
    
    in_zero_sequence = False
    zero_start_index = None
    
    for i in range(len(y)-1):
        if y[i] == 0:
            if not in_zero_sequence:
                in_zero_sequence = True
                zero_start_index = i
            else:
                y[i] = None
        else:
            if in_zero_sequence:
                in_zero_sequence = False
    
    return y, timestamps

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='plot_ops.py',
        description='Use glljobstat logs to plot data about the job',
        epilog='Example: ./plot_ops.py 1234 # For jobid 1234'
    )
    
    parser.add_argument(
        'jobid', type=str,
        help='Job ID to plot',
    )

    FLATTEN_CURVE = False
    
    args = parser.parse_args()
    if args.jobid == "a2424":
        job_detail = "\n Ran on small dataset for 5 hours. With singularity container, imsize=10000 with tier1-minimal.cfg \n With return line 1317 in pipeline.py"
    else:
        job_detail = "\n Ran on big dataset until completed. With singularity container, imsize=20000 with tier1-minimal.cfg \n With return line 1317 in pipeline.py"
    
    jobid = args.jobid

    regexp_timestamp = re.compile(r"timestamp: (\d+)")
    regexp_hist = re.compile(r'\"(' + str(jobid) + r':0:0)\":\s*\{ops: (\d+),.*?rb: (\{.*?\}), wb: (\{.*?\})')
    regexp_read = re.compile(r'"(' + str(jobid) + r':0:0)"\s*:\s*\{[^}]*\brb:\s*(\d+)')
    regexp_write= re.compile(r'"(' + str(jobid) + r':0:0)"\s*:\s*\{[^}]*\bwb:\s*(\d+)')
    regexp_operations = re.compile(r'\"(' + str(jobid) + r':0:0)\":\s*\{ops: (\d+)')
    regexp_read_rate = re.compile(r'"(' + str(jobid) + r':0:0)"\s*:\s*\{[^}]*\brd:\s*(\d+)')
    regexp_write_rate = re.compile(r'"(' + str(jobid) + r':0:0)"\s*:\s*\{[^}]*\bwr:\s*(\d+)')

    operations_metadata_labels = ['cr', 'op', 'cl', 'mn', 'ln', 'ul', 'mk', 'rm', 'mv',
                  'ga', 'sa', 'gx', 'sx', 'st', 'sy', 'pu', 'mi', 'fa', 
                  'dt', 'gi', 'si', 'qc', 'pa']
    
    operations_metadata_full_names = ['create', 'open', 'close', 'mknod', 'link', 'unlink', 'mkdir', 
        'rmdir', 'rename', 'getattr', 'setattr', 'getxattr', 'setxattr', 
        'statfs', 'sync', 'punch', 'migrate', 'fallocate', 
        'destroy', 'get_info', 'set_info', 'quotactl', 'prealloc', 
    ]
    print(len(operations_metadata_labels), len(operations_metadata_full_names))

    
    regexp_metadata_list =[]
    for operation in operations_metadata_labels:
        regexp_metadata_list.append(re.compile(r'\"(' + str(jobid) + r':0:0)\"\s*:\s*\{[^}]*\b' + operation + r':\s*(\d+)'))

    filename_rate = "log_" + str(jobid) + "_rate.log"
    filename_hist = "log_" + str(jobid) + "_hist.log"

    # DATA

    timestamps = []

    # Dictionnaries of read and write bytes per size operation
    read_bytes_hist_dict_list = []
    write_bytes_hist_dict_list = []

    # Read and write bytes size in GB, cumulative
    read_bytes_size = []
    write_bytes_size = []

    # Total operations cumulative
    total_operations = []

    # Metadata operations, cumulative
    metadata_operations_total = [[] for _ in operations_metadata_labels]

    # For plotting with different operation size, we use a gradient of color to differentiate the size of the operation and make the plot more readable
    operation_sizes = ["32", "64", "128", "256", "512", "1K", "2K", "4K", "8K", "16K","32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M","16M", "32M", "64M"]
    
    # For color gradient
    size_to_index = {size: i for i, size in enumerate(operation_sizes)}
    num_colors = len(operation_sizes)
    
    colors = [
        "#4b0082",  # Dark Indigo (32)
        "#2e8b57",  # Sea Green (64)
        "#6a5acd",  # Pastel Blue (128)
        "#a52a2a",  # Warm Brown (256)
        "#d2691e",  # Soft Brown (512)
        "#8b0000",  # Dark Red (1K)
        "#000000",  # Black (2K)
        "#6a5acd",  # Pastel Blue (4K)
        "#6495ed",  # Light Blue (8K)
        "#4682b4",  # Steel Blue (16K)
        "#40e0d0",  # Soft Turquoise (32K)
        "#2e8b57",  # Sea Green (64K)
        "#3cb371",  # Mint Green (128K)
        "#9acd32",  # Olive Yellow-Green (256K)
        "#b2d732",  # Lighter Lemon Green (512K)
        "#d4e157",  # Bright Yellow-Green (1M)
        "#ffd700",  # Golden Yellow (2M)
        "#ffcc66",  # Pastel Orange-Yellow (4M)
        "#ffb347",  # Light Orange (8M)
        "#ff9966",  # Salmon Orange (16M)
        "#ff6347",  # Soft Tomato Red (32M)
        "#ff4500",  # Dark Orange-Red (64M)
    ]


    # Assign a color to each size operation
    size_to_color = {size: colors[i % len(colors)] for i, size in enumerate(operation_sizes)}

    # open hist file

    first_iteration = True
    with open(filename_hist) as file:
        lines = file.read().split("---")
        for entry in lines:
            timestamp_match = regexp_timestamp.search(entry)
            if timestamp_match:
                timestamp = int(regexp_timestamp.search(entry).group(1))
                timestamps.append(timestamp)
                hist_match = regexp_hist.search(entry)
                
                if hist_match:

                    if first_iteration:
                        first_iteration = False
                        max_rb = eval(hist_match.group(3))
                        max_wb = eval(hist_match.group(4))

                    rb_dict = eval(hist_match.group(3))
                    wb_dict = eval(hist_match.group(4))

                    read_bytes_hist_dict_list.append(rb_dict)
                    write_bytes_hist_dict_list.append(wb_dict)

                    read_bytes_size.append(sum(convert_size(k) * v for k, v in rb_dict.items()))
                    write_bytes_size.append(sum(convert_size(k) * v for k, v in wb_dict.items()))

                    if read_bytes_size[-1] > sum(convert_size(k) * v for k, v in max_rb.items()):
                        max_rb = rb_dict
                    if write_bytes_size[-1] > sum(convert_size(k) * v for k, v in max_wb.items()):
                        max_wb = wb_dict
                else:
                    read_bytes_hist_dict_list.append({})
                    write_bytes_hist_dict_list.append({})

                    read_bytes_size.append(0)
                    write_bytes_size.append(0)
                
                ops_match = regexp_operations.search(entry)
                if ops_match:
                    total_operations.append(int(ops_match.group(2)))
                else:
                    total_operations.append(0)

                for i, metadata_match in enumerate(regexp_metadata_list):
                    metadata_match = metadata_match.search(entry)
                    if metadata_match:
                        metadata_operations_total[i].append(int(metadata_match.group(2)))
                    else:
                        metadata_operations_total[i].append(0)
        
        # Find indices where metadata_operations_total[i] has non-zero sum
        valid_indices = [i for i in range(len(metadata_operations_total)) if sum(metadata_operations_total[i]) > 0]

        # Use valid indices to filter both lists consistently
        operations_metadata_labels = [operations_metadata_labels[i] for i in valid_indices]
        operations_metadata_total = [metadata_operations_total[i] for i in valid_indices]
        operations_metadata_full_names = [operations_metadata_full_names[i] for i in valid_indices]
        print(len(operations_metadata_labels), len(operations_metadata_total))
        # Relative timestamps
        timestamps = [t - timestamps[0] for t in timestamps]


        ## Not necessary anymore
        if FLATTEN_CURVE:
            for i in range(1, len(read_bytes_size)):
                if read_bytes_size[i] < read_bytes_size[i-1]:
                    read_bytes_size[i] = read_bytes_size[i-1]
            for i in range(1, len(write_bytes_size)):
                if write_bytes_size[i] < write_bytes_size[i-1]:
                    write_bytes_size[i] = write_bytes_size[i-1]

        read_bytes_size, timestamps_read_total = filter_zero_intervals_with_none(read_bytes_size, timestamps)
        write_bytes_size, timestamps_write_total = filter_zero_intervals_with_none(write_bytes_size, timestamps)

        # Calculate delta read, summing up the values by converting the keys to GB
        delta_read = [sum(convert_size(k) * v for k, v in d.items()) for d in read_bytes_hist_dict_list]
        delta_read = [0] + [(delta_read[i] - delta_read[i-1])/10 for i in range(1, len(delta_read))]

        plt.figure(figsize=(10, 5))
        # plot only lines not the points
        plt.plot(timestamps, delta_read, label="Delta read", linestyle='-')
        plt.xlabel("Time in sec")
        plt.ylabel("Delta read in GB/s")
        title = "Delta read over time, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_delta_read.png")
        plt.savefig("pic/" + str(jobid) + "_delta_read.png")

        # Calculate delta write, summing up the values by converting the keys to GB
        delta_write = [sum(convert_size(k) * v for k, v in d.items()) for d in write_bytes_hist_dict_list]
        delta_write = [0] + [(delta_write[i] - delta_write[i-1])/10 for i in range(1, len(delta_write))]

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, delta_write, label="Delta write", linestyle='-')
        plt.xlabel("Time in sec")
        plt.ylabel("Delta write in GB/s")
        title = "Delta write over time, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_delta_write.png")
        plt.savefig("pic/" + str(jobid) + "_delta_write.png")

        # Plot read bytes
        plot_lines = []
        plt.figure(figsize=(10, 10))
        for key in set().union(*(d.keys() for d in read_bytes_hist_dict_list)):
            read_values = [d.get(key, 0) for d in read_bytes_hist_dict_list]
            color = get_color(key, size_to_color)
            line, = plt.plot(timestamps_read_total, read_values, label=key + " read", linestyle='-', color = color, linewidth=3)
            plot_lines.append((key, line))
            last_nonzero_idx = max(i for i, v in enumerate(read_values) if v > 0)
            last_x = timestamps_read_total[last_nonzero_idx]
            last_y = read_values[last_nonzero_idx]
            
            plt.text(last_x + 500, last_y, key, fontsize=10, color='black', 
                    verticalalignment='center', fontweight='bold')
        plt.xlabel("Time in sec")
        plt.ylabel("Read bytes count")
        title = "Read bytes over time per size operation, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plot_lines_sorted = sorted(plot_lines, key=lambda x: operation_sizes.index(x[0]) if x[0] in operation_sizes else float('inf'))
        sorted_lines = [line for _, line in plot_lines_sorted]
        sorted_labels = [key + " read" for key, _ in plot_lines_sorted] 
        plt.legend(sorted_lines, sorted_labels)
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_read_bytes.png")
        plt.savefig("pic/" + str(jobid) + "_read_bytes.png")

        # Plot write bytes
        plot_lines = []
        plt.figure(figsize=(10, 10))
        for key in set().union(*(d.keys() for d in write_bytes_hist_dict_list)):
            color = get_color(key, size_to_color)
            write_values = [d.get(key, 0) for d in write_bytes_hist_dict_list]
            line, = plt.plot(timestamps_read_total, write_values, label=key + " write", marker='o', linestyle='-', color=color)
            plot_lines.append((key, line))

            last_nonzero_idx = max(i for i, v in enumerate(write_values) if v > 0)
            last_x = timestamps_read_total[last_nonzero_idx]
            last_y = write_values[last_nonzero_idx]
            plt.text(last_x + 500, last_y, key, fontsize=10, color='black',
                    verticalalignment='center', fontweight='bold')
            
        plt.xlabel("Time in sec")
        plt.ylabel("Write bytes")
        title = "Write bytes over time per size operation, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)

        plot_lines_sorted = sorted(plot_lines, key=lambda x: operation_sizes.index(x[0]) if x[0] in operation_sizes else float('inf'))
        sorted_lines = [line for _, line in plot_lines_sorted]
        sorted_labels = [key + " write" for key, _ in plot_lines_sorted]

        plt.legend(sorted_lines, sorted_labels)
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_write_bytes.png")
        plt.savefig("pic/" + str(jobid) + "_write_bytes.png")

        # Plot total data transfer over time
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, read_bytes_size, label="Read bytes", linewidth=3, linestyle='-')
        plt.plot(timestamps, write_bytes_size, label="Write bytes", linewidth=3, linestyle='-')
        plt.xlabel("Time in sec")
        plt.ylabel("Data transfer in GB")
        title = "Total data transfer over time, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + "total_data_transfer_" + str(jobid) + ".png")
        plt.savefig("pic/" + str(jobid) + "_total_data_transfer.png")

        # Plot metadata operations over time
        plt.figure(figsize=(12, 5))
        print("len(operations_metadata_total)", len(operations_metadata_total))
        print("len(operations_metadata_labels)", len(operations_metadata_labels))
        for i in range(len(operations_metadata_total)):
            plt.plot(timestamps, operations_metadata_total[i], label=operations_metadata_full_names[i], linewidth=3, linestyle='-')
            if operations_metadata_full_names[i] == "open":
                # Text but with an horizontal offset 
                plt.text(timestamps[-1] + 600, operations_metadata_total[i][-1], "&"+ operations_metadata_labels[i], fontsize=10, color='black',
                    verticalalignment='center')
            else:
                plt.text(timestamps[-1], operations_metadata_total[i][-1], operations_metadata_labels[i], fontsize=10, color='black',
                    verticalalignment='center')
            
        # DEBUG
        # Print last values for every metadata operation
        for i in range(len(operations_metadata_total)):
            print(operations_metadata_full_names[i], operations_metadata_total[i][-1])

        plt.xlabel("Time in sec")
        plt.ylabel("Metadata operations count")
        title = "Metadata operations over time, DDF Pipeline, job " + str(jobid) + str(job_detail) + ". Close and open curves are slightly similar, which is why we can only see one line."
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_metadata_operations.png")
        plt.savefig("pic/" + str(jobid) + "_metadata_operations.png")


        # Plot total operations over time
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, total_operations, label="Total operations", linewidth=3, linestyle='-')
        plt.xlabel("Time in sec")
        plt.ylabel("Total operations count")
        title = "Total operations over time, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_total_operations.png")
        plt.savefig("pic/" + str(jobid) + "_total_operations.png")


        # Hist of read total data
        plt.figure(figsize=(10, 5))
        non_zero_read_operations = {k: v for k, v in max_rb.items() if v > 0}
        plt.bar(non_zero_read_operations.keys(), [convert_size(k) * v for k, v in non_zero_read_operations.items()])
        plt.xlabel("Size of read operation")
        plt.ylabel("Total read data transfer in GB")
        title = "Total read data transfer per size operation, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        print("Saving file: ", "pic/" + str(jobid) + "_read_hist.png")
        plt.savefig("pic/" + str(jobid) + "_read_hist.png")

        # Hist of write total data
        plt.figure(figsize=(10, 5))
        non_zero_write_operations = {k: v for k, v in max_wb.items() if v > 0}
        plt.bar(non_zero_write_operations.keys(), [convert_size(k) * v for k, v in non_zero_write_operations.items()])
        plt.xlabel("Size of write operation")
        plt.ylabel("Total write data transfer in GB")
        title = "Total write data transfer per size operation, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        print("Saving file: ", "pic/" + str(jobid) + "_write_hist.png")
        plt.savefig("pic/" + str(jobid) + "_write_hist.png")

        # Hist of write count data
        plt.figure(figsize=(10, 5))
        plt.bar(max_wb.keys(), [v for k, v in max_wb.items()])
        plt.xlabel("Size of write operation")
        plt.ylabel("Total write count")
        title = "Total write count per size operation, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        print("Saving file: ", "pic/" + str(jobid) + "_write_count_hist.png")
        plt.savefig("pic/" + str(jobid) + "_write_count_hist.png")

        # Hist of read count data
        plt.figure(figsize=(10, 5))
        plt.bar(max_rb.keys(), [v for k, v in max_rb.items()])
        plt.xlabel("Size of read operation")
        plt.ylabel("Total read count")
        title = "Total read count per size operation, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        print("Saving file: ", "pic/" + str(jobid) + "_read_count_hist.png")
        plt.savefig("pic/" + str(jobid) + "_read_count_hist.png")

        

        timestamps = []
        write_rate = []
        read_rate = []
        # Open rate file
        with open(filename_rate) as file:
            lines = file.read().split("---")
            for entry in lines:
                timestamp_match = regexp_timestamp.search(entry)
                if timestamp_match:
                    timestamp = int(regexp_timestamp.search(entry).group(1))
                    timestamps.append(timestamp)
                    read_match = regexp_read_rate.search(entry)
                    if read_match:
                        read_rate.append(int(read_match.group(2)))
                    else:
                        read_rate.append(0)

                    write_match = regexp_write_rate.search(entry)
                    if write_match:
                        write_rate.append(int(write_match.group(2)))
                    else:
                        write_rate.append(0)

        timestamps = [t - timestamps[0] for t in timestamps]

        # Plotting the total operation per seconds over the time
        total_write_operation = [sum(write_rate[:i]) for i in range(len(write_rate))]
        total_read_operation = [sum(read_rate[:i]) for i in range(len(read_rate))]

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, total_read_operation, label="Total read operation", linewidth=3, linestyle='-')
        plt.plot(timestamps, total_write_operation, label="Total write operation", linewidth=3, linestyle='-')
        plt.xlabel("Time in sec")
        plt.ylabel("Total operation count")
        title = "Total read and write operations over time, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_total_readwrite_ops.png")
        plt.savefig("pic/" + str(jobid) + "_total_readwrite_ops.png")

        read_rate, timestamps_read_rate = filter_zero_intervals_with_none(read_rate, timestamps)
        write_rate, timestamps_write_rate = filter_zero_intervals_with_none(write_rate, timestamps)

        # Plot read and write rate
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps_read_rate, read_rate, label="Read rate", linestyle='-')
        plt.plot(timestamps_write_rate, write_rate, label="Write rate", linestyle='-')
        plt.xlabel("Time in sec")
        plt.ylabel("Rates in Ops/s")
        title = "Operation rates over time, DDF Pipeline, job " + str(jobid) + str(job_detail)
        plt.title(title)
        plt.legend()
        plt.grid()
        print("Saving file: ", "pic/" + str(jobid) + "_ops_rate.png")
        plt.savefig("pic/" + str(jobid) + "_ops_rate.png")