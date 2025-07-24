import gzip
import json

def extract_flops(trace_file):
    total_flops = 0
    with gzip.open(trace_file, 'rt') as f:
        trace_data = json.load(f)
        for event in trace_data.get("traceEvents", []):
            args = event.get("args", {})
            if "flops" in args:
                total_flops += args["flops"]
    print(f"Total estimated FLOPs: {total_flops:,}")
    return total_flops

total_flops = extract_flops("logs/plugins/profile/2025.../local.trace.json.gz")
