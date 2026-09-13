"""Run every invariant test and summarise. Exits non-zero on any failure."""
import pathlib
import subprocess
import sys

# Fast invariant tests, then the slow one that actually trains models.
TESTS = ["test_recurrence.py", "test_decode_consistency.py", "test_ponder.py",
         "test_memory_locality.py", "test_memory_disk.py", "test_quant.py",
         "test_bandwidth.py"]
SLOW = ["test_learns.py"]          # trains 4 small models; minutes, not seconds

root = pathlib.Path(__file__).parent
tests = TESTS + ([] if "--fast" in sys.argv else SLOW)
fails = []
for t in tests:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")
    r = subprocess.run([sys.executable, "-W", "ignore", str(root / "tests" / t)],
                       capture_output=True, text=True)
    out = "\n".join(l for l in r.stdout.splitlines()
                    if "Warning" not in l and "Consider using" not in l)
    print(out)
    if r.returncode != 0:
        fails.append(t)
        print(r.stderr[-2000:])

print(f"\n{'=' * 70}")
if fails:
    print(f"FAILED: {', '.join(fails)}")
    sys.exit(1)
print(f"all {len(TESTS)} test modules passed")
