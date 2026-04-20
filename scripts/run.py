"""
run.py — unified entry point for the neural-processes-distribution-shift project.

Usage:
    uv run python scripts/run.py help
    uv run python scripts/run.py sweep [--z-dims none 16] [--clean] [--plots-only] [--no-train] [--no-bench] [--no-extra]
    uv run python scripts/run.py presentation
"""
import sys
import os
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


HELP_TEXT = """\
Neural Processes under Distribution Shift — run.py

Subcommands:
  sweep          Run the full experiment pipeline (train → benchmark → TTA → budget curves).
  presentation   Generate the PowerPoint slide deck (requires python-pptx dev dependency).
  help           Show this message.

Sweep flags:
  --z-dims none 16   Model variants to evaluate (none = deterministic TNP, 16 = Latent TNP).
  --clean            Wipe results/ before starting.
  --plots-only       Skip training; re-run benchmark + extras on existing weights.
  --no-train         Skip training phase only.
  --no-bench         Skip benchmarking phase only.
  --no-extra         Skip TTA visual and budget scripts.

Examples:
  uv run python scripts/run.py sweep --z-dims none 16 --clean
  uv run python scripts/run.py sweep --plots-only --z-dims none 16
  uv run python scripts/run.py presentation
"""


def cmd_sweep(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="run.py sweep")
    parser.add_argument("--no-train",    action="store_true")
    parser.add_argument("--no-bench",    action="store_true")
    parser.add_argument("--no-extra",    action="store_true")
    parser.add_argument("--plots-only",  action="store_true",
                        help="Skip training; re-run benchmark + extras on existing weights.")
    parser.add_argument("--z-dims", nargs="+", default=["none", "16"])
    parser.add_argument("--clean",       action="store_true")
    args = parser.parse_args(argv)

    if args.plots_only:
        args.no_train = True

    # Import sweep logic from sweep.py (same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    import sweep as _sweep

    if args.clean:
        import shutil
        if Path("results").exists():
            print(">>> [Cleanup] Deleting results/ ...")
            shutil.rmtree("results")

    z_dims = []
    for z in args.z_dims:
        z_dims.append(None if z.lower() == "none" else int(z))

    datasets       = ["gp", "sinusoid"]
    robust_flags   = [False, True]
    context_sizes  = [10, 20, 40]
    experiments    = [(ds, r, ctx)
                      for ctx in context_sizes
                      for ds  in datasets
                      for r   in robust_flags]

    if not args.no_train:
        _sweep.run_training_phase(experiments, z_dims)
    if not args.no_bench:
        _sweep.run_benchmarking_phase(experiments, z_dims)
    if not args.no_extra:
        import subprocess
        for z in z_dims:
            extra_args = ["--z-dim", str(z)] if z is not None else []
            print(f"\n>>> [Extra] TTA Prototypes (z_dim={z})")
            subprocess.run([sys.executable, str(Path(__file__).parent / "test_time_adapt.py")] + extra_args, check=True)
            print(f"\n>>> [Extra] TTA Budget Curves (z_dim={z})")
            subprocess.run([sys.executable, str(Path(__file__).parent / "tta_budget.py")] + extra_args, check=True)

    print("\nSweep complete.")


def cmd_presentation(_argv):
    try:
        import generate_presentation as _pres
        _pres.main()
    except ImportError:
        print("Error: python-pptx is not installed.")
        print("Install it with:  uv sync  (it is in the [dev] dependency group)")
        sys.exit(1)


def cmd_help(_argv):
    print(HELP_TEXT)


COMMANDS = {
    "sweep":        cmd_sweep,
    "presentation": cmd_presentation,
    "help":         cmd_help,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(HELP_TEXT)
        sys.exit(0 if len(sys.argv) < 2 else 1)

    subcommand = sys.argv[1]
    COMMANDS[subcommand](sys.argv[2:])


if __name__ == "__main__":
    main()
