"""
VECTOR Test Runner
Runs all evaluation scripts and writes results to a single UTF-8 report.
"""
import sys, os, io, contextlib

# Patch stdout to avoid encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(WORKSPACE, "Output_Artifacts", "Logs", "test_report.txt")

def run_script(path, label):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}\n")
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        # Each script uses __file__ — patch it
        ns = {"__file__": path, "__name__": "__main__"}
        exec(compile(code, path, "exec"), ns)
    except SystemExit:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")

with open(REPORT, "w", encoding="utf-8") as rep, \
     contextlib.redirect_stdout(rep), \
     contextlib.redirect_stderr(rep):

    sys.stdout = rep  # also capture exec print()

    run_script(os.path.join(WORKSPACE, "3_Experiments", "moneyvis_inference.py"),
               "TEST 1: MoneyVis Inference")

    run_script(os.path.join(WORKSPACE, "compare_distributions.py"),
               "TEST 2: Feature Distribution Comparison")

    run_script(os.path.join(WORKSPACE, "2_Model_Training", "analyse_results.py"),
               "TEST 3: Strategy Comparison & Threshold Analysis (5-Fold CV)")

print(f"Report written to: {REPORT}")
