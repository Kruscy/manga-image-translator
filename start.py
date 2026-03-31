import os
import sys
import subprocess
from pathlib import Path

# ── Beállítások ────────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Manga\input"
OUTPUT_DIR = r"C:\Manga\output"
USE_GPU    = True
# ───────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
CONFIG_DIR  = SCRIPT_DIR / "config"

def list_configs():
    return sorted(p for p in CONFIG_DIR.glob("*.json"))

def main():
    configs = list_configs()
    if not configs:
        print("Nem található .json config fájl a config/ mappában.")
        input("Nyomj Enter-t a kilépéshez...")
        sys.exit(1)

    print("=" * 52)
    print("  Manga Image Translator – Config választó")
    print("=" * 52)
    for i, path in enumerate(configs, 1):
        print(f"  {i:>2}.  {path.name}")
    print("-" * 52)

    while True:
        try:
            raw = input(f"Válassz config-ot (1-{len(configs)}): ").strip()
            choice = int(raw)
            if 1 <= choice <= len(configs):
                break
            print(f"  Kérlek 1 és {len(configs)} közötti számot adj meg.")
        except (ValueError, EOFError):
            print("  Érvénytelen bemenet, próbáld újra.")

    selected = configs[choice - 1]
    print(f"\nKiválasztva: {selected.name}")
    print(f"Bemenet:     {INPUT_DIR}")
    print(f"Kimenet:     {OUTPUT_DIR}")
    print("=" * 52)

    cmd = [
        sys.executable, "-m", "manga_translator",
        "local",
        "-i", INPUT_DIR,
        "-o", OUTPUT_DIR,
        "--config", str(selected),
    ]
    if USE_GPU:
        cmd.append("--use-gpu")

    print("Indítás...\n")
    try:
        subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    except KeyboardInterrupt:
        print("\nLeállítva.")

if __name__ == "__main__":
    main()
