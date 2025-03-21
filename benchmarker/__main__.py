import importlib.util
import argparse
import os
import time
import torch
import torch.distributed as dist
from benchmarker.methods.methods_manager import MethodManager
import importlib
import csv
from datetime import datetime
import shutil


metadata_dir = ".benchmarker_metadata"

def init_distributed():
    
    """Initialize distributed processing with MPI backend."""
    dist.init_process_group(backend="gloo")  # Or "gloo" for CPU-only

    world_size = dist.get_world_size()  # Total number of processes
    global_rank = dist.get_rank()  # Rank of current process
    ngpus = torch.cuda.device_count()
    local_rank = global_rank % ngpus if ngpus != 0 else 0 # Rank within node (GPU ID)

    return world_size, global_rank, local_rank

class Benchmarker:

    @classmethod
    def run_test(cls, class_name, method_args, rank, metadada):
        """Runs a test class instance with API parameters and tracks execution time."""
        
        
        args = dict(pair.split('=') for pair in method_args.split(',')) if method_args else {}
        test_instance = MethodManager.get_method(class_name)(**args)
        start_time = time.time()  # Start time
        test_instance.invoke()
        end_time = time.time()  # End time
        
        # Write data to CSV
        rank_file = os.path.join(metadada, f"{rank}.txt")
        with open(rank_file, "w") as f:
            execution_time = end_time - start_time
            execution_time_ms = round(execution_time * 1000, 3)
            execution_time = round(execution_time, 3)
            start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_time = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
            print(rank, start_time, end_time, execution_time, execution_time_ms)
            f.write(f"{rank},{start_time},{end_time},{execution_time},{execution_time_ms}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-plugin", type=str, default=None, help="Python file containing the method class.")
    parser.add_argument("--test-method", type=str, help="Method to run in parallel.")
    parser.add_argument("--method-args", type=str, help="Arguments for the method should be separated by commas. Example: --method_args='api_url=api-url,api_key=your-key,model=model-name'.")
    parser.add_argument("--output-path", type=str, default="test_results.csv", help="Path to save the result. Example: result.csv")
    parser.add_argument("--list-methods", action="store_true", help="List all available methods and exit.")

    
    # Initialize distributed processing
    _, global_rank, _ = init_distributed()

    args = parser.parse_args()

    # Ensure required arguments are provided unless --list-methods is used
    if not args.list_methods and (not args.test_method or not args.method_args):
        parser.error("--test-method and --method-args are required unless using --list-methods.")
    elif args.list_methods:
        if global_rank == 0:
            print()
            print("Available Methods:", list(MethodManager.list_methods()))
        return

    if args.method_plugin:
        MethodManager.import_method(args.method_plugin)

    # Start time tracking only for process 0
    
    if global_rank == 0:
        if os.path.exists(metadata_dir):
            shutil.rmtree(metadata_dir)  # Delete directory
        os.makedirs(metadata_dir, exist_ok=True)  # Recreate directory

    torch.distributed.barrier()

    start_time = time.time() if global_rank == 0 else None
    Benchmarker.run_test(args.test_method, args.method_args, global_rank, metadata_dir)

    torch.distributed.barrier()

    # Print process completion time
    if global_rank == 0:
        total_time = time.time() - start_time
        print(f"\nFinal total execution time for all processes: {total_time:.4f} sec")

    if global_rank == 0:
        with open(args.output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Rank", "Start Time", "End Time", "Execution Time (s)", "Execution Time (ms)"])

            # Collect and merge all rank files
            for rank_file in sorted(os.listdir(metadata_dir)):
                if rank_file.endswith(".txt"):
                    with open(os.path.join(metadata_dir, rank_file), "r") as rf:
                        for line in rf:
                            writer.writerow(line.strip().split(","))
        shutil.rmtree(metadata_dir)

if __name__ == "__main__":
    main()

