import os
import tarfile
import time
from pathlib import Path

def is_excluded(path, exclude_names, exclude_exts):
    name = path.name
    if name in exclude_names: return True
    if any(name.endswith(ext) for ext in exclude_exts): return True
    return False

def build_lite_bundle():
    worker_dir = Path(r"c:\Users\SSAFY\Desktop\new_focus_PJT\AI\worker").resolve()
    ai_dir = worker_dir.parent
    dist_dir = ai_dir / "dist"
    dist_dir.mkdir(exist_ok=True)
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"divery-gpu-worker-lite-{ts}.tar.gz"
    bundle_path = dist_dir / bundle_name
    
    print("========================================")
    print(" Divery GPU Worker Bundle (Lite) - Windows")
    print("========================================")
    
    with tarfile.open(bundle_path, "w:gz") as tar:
        # Add worker
        print("Adding worker directory...")
        for root, dirs, files in os.walk(worker_dir):
            dirs[:] = [d for d in dirs if not is_excluded(Path(d), {"__pycache__"}, [])]
            for f in files:
                # Exclude .env, worker.pid, worker.log, and .pyc files
                if is_excluded(Path(f), {".env", "worker.pid", "worker.log"}, {".pyc"}): continue
                file_path = Path(root) / f
                arc_name = file_path.relative_to(ai_dir)
                tar.add(file_path, arcname=f"AI/{arc_name.as_posix()}")
                
        # Add CFD_fishial
        print("Adding CFD_fishial directory (Excluding Weights!)...")
        cfd_dir = ai_dir / "CFD_fishial"
        for root, dirs, files in os.walk(cfd_dir):
            # Exclude specific folders
            dirs[:] = [d for d in dirs if not is_excluded(Path(d), {
                "__pycache__", "results", "test_videos", "fishial_model", "weights"
            }, [])]
            for f in files:
                # Exclude specific extensions related to weights
                if is_excluded(Path(f), {}, {".pyc", ".pt", ".pth", ".ckpt"}): continue
                file_path = Path(root) / f
                arc_name = file_path.relative_to(ai_dir)
                tar.add(file_path, arcname=f"AI/{arc_name.as_posix()}")

        # Add README
        print("Adding README.md...")
        readme_path = ai_dir / "README.md"
        if readme_path.exists():
            tar.add(readme_path, arcname="AI/README.md")
            
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    print(f"\n[Success] Bundle Created!")
    print(f"Path: {bundle_path}")
    print(f"Size: {size_mb:.2f} MB")
    print("========================================")

if __name__ == "__main__":
    build_lite_bundle()
