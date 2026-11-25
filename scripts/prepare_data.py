import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate retail training data")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2] / "shelfveil_local")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    from shelfveil.data import prepare_data

    try:
        result = prepare_data(args.workspace, args.threads, args.batch_size)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Preparation stopped: {error}\n")
    print(f"Series: {result['series']}")
    print(f"Daily rows: {result['daily_rows']}")
    print(f"Accessible hourly observations: {result['hourly_rows']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    print(f"Peak process memory MiB: {result['peak_process_memory_bytes'] / 1024**2:.2f}")


if __name__ == "__main__":
    main()
