import re
import argparse

def parse_size(size_str):
    """Convert strings like '54.45 Gb' or '869.21 Mb' to bytes."""
    size_str = size_str.strip()
    if size_str == '--' or size_str.lower() in ('', '0 b'):
        return 0
    num, unit = size_str.split()
    num = float(num)
    unit = unit.lower()
    if unit == 'kb':
        return num * 1e3
    elif unit == 'mb':
        return num * 1e6
    elif unit == 'gb':
        return num * 1e9
    elif unit == 'b':
        return num
    else:
        raise ValueError(f"Unknown unit in memory size: {size_str}")

def main(args):
    with open(args.logfile, 'r') as f:
        lines = f.readlines()

    flops_total = 0.0
    mem_total_bytes = 0

    # Identify table header
    start_index = None
    for i, line in enumerate(lines):
        if "Name" in line and "Total GFLOPs" in line:
            start_index = i + 2  # skip dashed line too
            break
    if start_index is None:
        raise ValueError("Profiler table header not found.")

    # Parse each line
    for line in lines[start_index:]:
        columns = re.split(r'\s{2,}', line.strip())
        if len(columns) < 9:
            continue
        try:
            flops = float(columns[9])
            flops_total += flops
        except ValueError:
            pass  # skip if no GFLOPs value

        try:
            mem_bytes = parse_size(columns[6])  # Self CPU Mem
            mem_total_bytes += mem_bytes
        except Exception:
            pass  # in case column is '--' or missing

    print(f"Total TFLOPs: {flops_total / 1e3:.2f}")
    print(f"Total Memory Volume: {mem_total_bytes / 1e9:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PyTorch profiler log for total GFLOPs and memory volume.")
    parser.add_argument('--logfile', type=str, required=True, help='Path to the PyTorch profiler log file.')
    args = parser.parse_args()

    main(args)