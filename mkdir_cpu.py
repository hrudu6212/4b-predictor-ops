from pathlib import Path
p = Path("/tmp/4b_work")
p.mkdir(parents=True, exist_ok=True)
Path("/tmp/4b_data/out").mkdir(parents=True, exist_ok=True)
print("dirs ready", flush=True)
