"""Run the test and save results."""
import subprocess, sys
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_data.py", "--tb=line", "-v"],
    capture_output=True, text=True, timeout=120
)
with open("pytest_final.txt", "w") as f:
    f.write(r.stdout + "\n---STDERR---\n" + r.stderr)
print(r.stdout[-500:])
print(r.stderr[-500:])
print("Exit:", r.returncode)