#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import pathlib

STUDIO_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR = STUDIO_DIR / "outputs"


def print_banner():
    print("\n" + "=" * 62)
    print("      🎬 SMART STUDIO · WAN 2.2 & HERMES 3 LLM RUNNER      ")
    print("=" * 62)
    print("  [1] Start Smart Studio frontend locally (localhost:7860)")
    print("  [2] Deploy/Redeploy Smart Studio backend to Modal (Wan + Hermes)")
    print("  [3] Pre-cache ALL model weights in Modal volume (Wan 2.2 + Hermes 3)")
    print("  [4] Pre-cache Wan 2.2 model weights on Modal")
    print("  [5] Pre-cache Hermes 3 model weights on Modal")
    print("  [6] Start frontend on Local Network (Mobile / Tablet access)")
    print("  [7] Run Smart Studio automated test suite")
    print("  [8] Clear Smart Studio outputs folder (outputs/)")
    print("  [9] Reconnect/Setup Modal account (runs 'modal setup')")
    print("  [0] Exit")
    print("=" * 62)


def run_cmd(cmd, cwd=STUDIO_DIR):
    try:
        res = subprocess.run(cmd, cwd=str(cwd), shell=True)
        return res.returncode
    except KeyboardInterrupt:
        print("\n[Smart Studio] Interrupted by user.")
        return 0


def clear_outputs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in list(OUTPUTS_DIR.iterdir()):
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except Exception as e:
                print(f"Error removing {f.name}: {e}")
    print(f"\n✓ Cleared {count} file(s) from Smart Studio outputs folder.")


def main():
    os.chdir(str(STUDIO_DIR))
    while True:
        print_banner()
        choice = input("\nEnter choice [0-9] (default 1): ").strip()
        if not choice:
            choice = "1"

        if choice == "1":
            print("\n---> Launching Smart Studio on localhost:7860...")
            run_cmd(f'"{sys.executable}" app.py')
        elif choice == "2":
            print("\n---> Deploying Smart Studio backend to Modal...")
            run_cmd(f'"{sys.executable}" -m modal deploy deploy_smart_wan.py')
        elif choice == "3":
            print("\n---> Pre-caching ALL model weights in Modal persistent volume...")
            run_cmd(f'"{sys.executable}" -m modal run deploy_smart_wan.py::download_all_smart_weights')
        elif choice == "4":
            print("\n---> Pre-caching Wan 2.2 model weights in Modal persistent volume...")
            run_cmd(f'"{sys.executable}" -m modal run deploy_smart_wan.py::download_wan_weights')
        elif choice == "5":
            print("\n---> Pre-caching Hermes 3 model weights in Modal persistent volume...")
            run_cmd(f'"{sys.executable}" -m modal run deploy_smart_wan.py::download_hermes_weights')
        elif choice == "6":
            print("\n---> Launching Smart Studio with local network / mobile access...")
            env = os.environ.copy()
            env["GRADIO_SERVER_NAME"] = "0.0.0.0"
            subprocess.run(f'"{sys.executable}" app.py', cwd=str(STUDIO_DIR), env=env, shell=True)
        elif choice == "7":
            print("\n---> Running Smart Studio automated test suite...")
            run_cmd(f'"{sys.executable}" -m unittest discover tests')
        elif choice == "8":
            clear_outputs()
        elif choice == "9":
            print("\n---> Running 'modal setup'...")
            run_cmd(f'"{sys.executable}" -m modal setup')
        elif choice == "0":
            print("\nExiting Smart Studio. Goodbye!")
            break
        else:
            print(f"\nInvalid option '{choice}'. Please enter 0-9.")


if __name__ == "__main__":
    main()
