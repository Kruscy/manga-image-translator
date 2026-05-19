import os
import sys
import subprocess
from pathlib import Path

# ── Beállítások ────────────────────────────────────────────────────────────────
HOST     = "0.0.0.0"   # 0.0.0.0 = külső gépek is elérhetik
PORT     = 5003
USE_GPU  = True
WORKERS  = 2           # Kétszakaszos pipeline-hoz 2 worker elegendő (GPU csak stage1+stage2 alatt terhelt)
# ───────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

def main():
    print("=" * 52)
    print("  Manga Image Translator – API Szerver")
    print("=" * 52)
    print(f"  Host:    {HOST}")
    print(f"  Port:    {PORT}")
    print(f"  GPU:     {USE_GPU}")
    print(f"  Workers: {WORKERS} párhuzamos fordító")
    print(f"  Config:  config/gpt_web.json (alapértelmezett)")
    print("=" * 52)
    print("Szerver indítása...")
    print(f"API docs: http://[PC-IP]:{PORT}/docs\n")

    cmd = [
        sys.executable,
        "server/main.py",
        "--host", HOST,
        "--port", str(PORT),
        "--start-instance",
        "--workers", str(WORKERS),
    ]
    if USE_GPU:
        cmd.append("--use-gpu")

    try:
        subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    except KeyboardInterrupt:
        print("\nSzerver leállítva.")

if __name__ == "__main__":
    main()
