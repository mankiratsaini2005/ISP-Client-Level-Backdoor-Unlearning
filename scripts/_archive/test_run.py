import subprocess
try:
    result = subprocess.run(
        ["./.venv/bin/python", "scripts/run_federated.py", "--config", "configs/stage7c_d1_attacked_canonical.yaml", "--mode", "main"],
        env={"PYTHONPATH": "."},
        capture_output=True,
        text=True,
        timeout=10
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
except subprocess.TimeoutExpired as e:
    print("Timeout! STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
