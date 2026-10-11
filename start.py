import os
import sys
import subprocess
import webbrowser
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def install_package(package_name, pip_name=None):
    pip_name = pip_name or package_name
    print(f"Installing {package_name} package locally...")
    subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)


def ensure_modal_installed():
    try:
        import modal  # noqa: F401
    except ImportError:
        install_package("modal")


def deploy_wan():
    ensure_modal_installed()
    print("\n---> Deploying Wan 2.2 Video backend (deploy_wan.py) to Modal...")
    subprocess.run([sys.executable, "-m", "modal", "deploy", "deploy_wan.py"], check=True)
    print("✓ Wan 2.2 backend deployed to Modal!")


def deploy_qwen():
    ensure_modal_installed()
    print("\n---> Deploying Qwen Image Edit backend (deploy_qwen.py) to Modal...")
    subprocess.run([sys.executable, "-m", "modal", "deploy", "deploy_qwen.py"], check=True)
    print("✓ Qwen Image Edit backend deployed to Modal!")


def deploy_z_image():
    ensure_modal_installed()
    print("\n---> Deploying Z-Image Turbo Character backend (deploy_z_image.py) to Modal...")
    subprocess.run([sys.executable, "-m", "modal", "deploy", "deploy_z_image.py"], check=True)
    print("✓ Z-Image Turbo backend deployed to Modal!")


def deploy_all():
    ensure_modal_installed()
    print("\n" + "=" * 60)
    print(" Deploying All Wan 2.2 AI Studio Backends to Modal ".center(60, "="))
    print("=" * 60)
    deploy_wan()
    deploy_qwen()
    deploy_z_image()
    print("\n" + "=" * 60)
    print(" ✓ ALL 3 STUDIO BACKENDS DEPLOYED SUCCESSFULLY! ".center(60, "="))
    print("=" * 60)


def cache_wan_weights():
    ensure_modal_installed()
    print("\n---> Caching Wan 2.2 model weights in Modal volume...")
    subprocess.run([sys.executable, "-m", "modal", "run", "deploy_wan.py::download_all_weights"], check=True)
    print("✓ Wan 2.2 model weights cached successfully!")


def cache_qwen_weights():
    ensure_modal_installed()
    print("\n---> Caching Qwen Image Edit model weights in Modal volume...")
    subprocess.run([sys.executable, "-m", "modal", "run", "deploy_qwen.py::download_qwen_weights"], check=True)
    print("✓ Qwen Image Edit model weights cached successfully!")


def cache_z_image_weights():
    ensure_modal_installed()
    print("\n---> Caching Z-Image Turbo model weights in Modal volume...")
    subprocess.run([sys.executable, "-m", "modal", "run", "deploy_z_image.py::download_z_image_weights"], check=True)
    print("✓ Z-Image Turbo model weights cached successfully!")


def cache_all_weights():
    cache_wan_weights()
    cache_qwen_weights()
    cache_z_image_weights()


def clear_outputs(script_dir: str):
    print("\n---> Clearing local outputs folder...")
    outputs_dir = os.path.join(script_dir, "outputs")
    if os.path.exists(outputs_dir):
        count = 0
        for root, dirs, files in os.walk(outputs_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                    count += 1
                except Exception as e:
                    print(f"Failed to delete {name}: {e}")
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                    count += 1
                except Exception as e:
                    print(f"Failed to delete {name}: {e}")
        os.makedirs(outputs_dir, exist_ok=True)
        print(f"\n✓ Cleared {count} item(s) from outputs folder.")
    else:
        os.makedirs(outputs_dir, exist_ok=True)
        print("\n✓ Outputs folder was already empty.")


def manage_modal_profiles():
    ensure_modal_installed()
    print("\n" + "=" * 55)
    print("         MODAL ACCOUNT & PROFILE MANAGER         ")
    print("=" * 55)
    subprocess.run([sys.executable, "-m", "modal", "profile", "list"])
    print("\nActions:")
    print("  [1] Switch to an existing profile (activate account)")
    print("  [2] Log into / set up a new Modal account ('modal setup')")
    print("  [0] Back to main menu")
    sub_choice = input("\nEnter choice [0-2] (default 1): ").strip()
    if sub_choice == "1" or sub_choice == "":
        profile_name = input("Enter profile/workspace name to activate: ").strip()
        if profile_name:
            res = subprocess.run([sys.executable, "-m", "modal", "profile", "activate", profile_name])
            if res.returncode == 0:
                print(f"\n✓ Switched active Modal account to: '{profile_name}'")
                print("  Note: Re-deploy backends to this account if needed.")
            else:
                print(f"\n❌ Could not activate profile '{profile_name}'. Check spelling.")
    elif sub_choice == "2":
        print("\n---> Running 'modal setup' in browser...")
        subprocess.run([sys.executable, "-m", "modal", "setup"])


def launch_frontend():
    print("\n---> Checking local dependencies...")

    try:
        import gradio  # noqa: F401
    except ImportError:
        install_package("gradio", "gradio>=5.0,<7.0")

    ensure_modal_installed()

    try:
        import cv2  # noqa: F401
    except ImportError:
        install_package("opencv-python")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        install_package("pillow")

    try:
        import numpy  # noqa: F401
    except ImportError:
        install_package("numpy")

    try:
        import requests  # noqa: F401
    except ImportError:
        install_package("requests")

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    if os.environ.get("WAN_SHARE_LAN") == "1":
        print(f"\n---> Starting local Gradio app on local network: http://{local_ip}:7860...")
        print(f"👉 To access from your mobile phone, open: http://{local_ip}:7860")
    else:
        print("\n---> Starting local Gradio app on http://localhost:7860...")

    try:
        cmd = [sys.executable, "app.py"]

        def open_browser():
            time.sleep(3)
            print("\nOpening http://localhost:7860 in your default browser...")
            webbrowser.open("http://localhost:7860")

        import threading
        t = threading.Thread(target=open_browser)
        t.daemon = True
        t.start()

        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping local server...")
    except Exception as e:
        print(f"\n❌ Failed to run Gradio app: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Support non-interactive CLI flags from start.bat or terminal
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        try:
            if arg in ("--deploy-all", "deploy-all"):
                deploy_all()
                return
            elif arg in ("--deploy-wan", "deploy-wan"):
                deploy_wan()
                return
            elif arg in ("--deploy-qwen", "deploy-qwen"):
                deploy_qwen()
                return
            elif arg in ("--deploy-z-image", "deploy-z-image"):
                deploy_z_image()
                return
            elif arg in ("--cache-all", "cache-all"):
                cache_all_weights()
                return
            elif arg in ("--cache-wan", "cache-wan"):
                cache_wan_weights()
                return
            elif arg in ("--cache-qwen", "cache-qwen"):
                cache_qwen_weights()
                return
            elif arg in ("--cache-z-image", "cache-z-image"):
                cache_z_image_weights()
                return
            elif arg in ("--profile", "profile"):
                manage_modal_profiles()
                return
            elif arg in ("--clear-outputs", "clear-outputs"):
                clear_outputs(script_dir)
                return
            elif arg in ("--lan", "lan"):
                os.environ["WAN_SHARE_LAN"] = "1"
                launch_frontend()
                return
            elif arg in ("--run", "run"):
                launch_frontend()
                return
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Operation failed: {e}")
            sys.exit(1)

    print("=" * 60)
    print(" Wan 2.2 Studio Localhost Runner ".center(60, "="))
    print("=" * 60)

    print("[Options]")
    print("1. Start Studio frontend locally")
    print("2. Deploy/Redeploy Wan 2.2 backend to Modal")
    print("3. Deploy/Redeploy Qwen Image Edit backend to Modal")
    print("4. Pre-cache Wan 2.2 model weights on Modal")
    print("5. Pre-cache Qwen Image Edit model weights on Modal")
    print("6. Start frontend on Local Network (mobile access)")
    print("7. Switch / Manage Modal Account Profiles (runs 'modal profile' / 'modal setup')")
    print("8. Clear local outputs folder (outputs/)")
    print("9. Deploy/Redeploy Z-Image Turbo Character backend to Modal")
    print("10. Deploy ALL Studio Backends to Modal (Wan 2.2 + Qwen + Z-Image Turbo)")

    choice = input("\nEnter choice [1-10] (default 1): ").strip()

    try:
        if choice == "2":
            deploy_wan()
            sys.exit(0)
        elif choice == "3":
            deploy_qwen()
            sys.exit(0)
        elif choice == "4":
            cache_wan_weights()
            sys.exit(0)
        elif choice == "5":
            cache_qwen_weights()
            sys.exit(0)
        elif choice == "6":
            os.environ["WAN_SHARE_LAN"] = "1"
        elif choice == "7":
            manage_modal_profiles()
            sys.exit(0)
        elif choice == "8":
            clear_outputs(script_dir)
            sys.exit(0)
        elif choice == "9":
            deploy_z_image()
            sys.exit(0)
        elif choice == "10":
            deploy_all()
            sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment/Operation failed: {e}")
        sys.exit(1)

    launch_frontend()


if __name__ == "__main__":
    main()
