#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import pathlib

STUDIO_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR = STUDIO_DIR / "outputs"


def get_current_modal_profile():
    try:
        res = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip() and "DeprecationWarning" not in line]
        return lines[-1] if lines else "unknown"
    except Exception:
        return "unknown"


def print_banner():
    active_profile = get_current_modal_profile()
    print("\n" + "=" * 64)
    print("      🎬 SMART STUDIO · WAN 2.2 & HERMES 3 LLM RUNNER      ")
    print("=" * 64)
    print(f"  Active Modal Account/Profile: [{active_profile}]")
    print("-" * 64)
    print("  [1] Start Smart Studio frontend locally (localhost:7860)")
    print("  [2] Deploy/Redeploy Smart Studio backend to Modal (Wan + Hermes)")
    print("  [3] Pre-cache ALL model weights in Modal volume (Wan 2.2 + Hermes 3)")
    print("  [4] Pre-cache Wan 2.2 model weights on Modal")
    print("  [5] Pre-cache Hermes 3 model weights on Modal")
    print("  [6] Start frontend on Local Network (Mobile / Tablet access)")
    print("  [7] Run Smart Studio automated test suite")
    print("  [8] Clear Smart Studio outputs folder (outputs/)")
    print("  [9] Switch / Manage Modal Account Profiles (change account)")
    print("  [0] Exit")
    print("=" * 64)


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


def manage_modal_profiles():
    print("\n" + "=" * 55)
    print("         MODAL ACCOUNT & PROFILE MANAGER         ")
    print("=" * 55)
    subprocess.run([sys.executable, "-m", "modal", "profile", "list"])
    print("\nActions:")
    print("  [1] Switch to an existing profile (activate account)")
    print("  [2] Log into / set up a new Modal account ('modal setup')")
    print("  [0] Back to main menu")
    choice = input("\nEnter choice [0-2] (default 1): ").strip()
    if choice == "1" or choice == "":
        profile_name = input("Enter profile/workspace name to activate: ").strip()
        if profile_name:
            res = subprocess.run([sys.executable, "-m", "modal", "profile", "activate", profile_name])
            if res.returncode == 0:
                print(f"\n✓ Switched active Modal account to: '{profile_name}'")
                print("  Note: Remember to run Option 2 to deploy backend to this account if needed.")
            else:
                print(f"\n❌ Could not activate profile '{profile_name}'. Check spelling.")
    elif choice == "2":
        print("\n---> Running 'modal setup' in browser...")
        subprocess.run([sys.executable, "-m", "modal", "setup"])


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
            manage_modal_profiles()
        elif choice == "0":
            print("\nExiting Smart Studio. Goodbye!")
            break
        else:
            print(f"\nInvalid option '{choice}'. Please enter 0-9.")


if __name__ == "__main__":
    main()
