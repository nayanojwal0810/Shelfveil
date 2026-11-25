import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Recover historical demand and evaluate controlled stockouts")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2] / "shelfveil_local")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    from shelfveil.recovery import run_recovery

    try:
        result = run_recovery(args.workspace, args.threads, args.batch_size)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Recovery stopped: {error}\n")
    print(f"Historical daily rows: {result['historical_rows']}")
    print(f"Controlled mask cases: {result['mask_cases']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    print(f"Peak process memory MiB: {result['peak_process_memory_bytes'] / 1024**2:.2f}")


if __name__ == "__main__":
    main()
