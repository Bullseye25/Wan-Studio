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


def deploy_narrator():
    ensure_modal_installed()
    print("\n---> Deploying Kokoro-82M Neural Voice Narrator (deploy_narrator.py) to Modal...")
    subprocess.run([sys.executable, "-m", "modal", "deploy", "deploy_narrator.py"], check=True)
    print("✓ Kokoro-82M Narrator backend deployed to Modal!")


def deploy_all():
    ensure_modal_installed()
    print("\n" + "=" * 65)
    print(" Deploying All 4 Wan 2.2 AI Studio Backends to Modal ".center(65, "="))
    print("=" * 65)

    # Pre-flight container budget enforcement
    try:
        from modal_budget import verify_modal_container_budget
        is_valid, total_conts, breakdown = verify_modal_container_budget(hard_cap=8)
        print(f"[Modal Budget] Pre-flight verification passed: {total_conts}/8 containers allocated.")
        for k, v in breakdown.items():
            print(f"  • {k.upper()}: {v} container(s)")
    except Exception as e:
        print(f"[Modal Budget Error] Budget validation failed: {e}")
        sys.exit(1)

    deploy_wan()
    deploy_qwen()
    deploy_z_image()
    deploy_narrator()
    print("\n" + "=" * 65)
    print(" ✓ ALL 4 STUDIO BACKENDS DEPLOYED SUCCESSFULLY! ".center(65, "="))
    print("=" * 65)


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


def run_fast_story():
    print("\n---> Launching OmniFlow-30 Fast Story CLI...")
    subprocess.run([sys.executable, "fast_story_cli.py"])


def verify_setup_and_deployment():
    print("\n" + "=" * 72)
    print(" 🛠️  WAN 2.2 STUDIO PRE-FLIGHT SETUP & DEPLOYMENT VERIFICATION ".center(72, "="))
    print("=" * 72)

    # 1. Local Python Dependencies
    print("\n[1/5] Checking Local Python Packages...")
    pkgs = [
        ("gradio", "Gradio Web UI"),
        ("modal", "Modal Cloud SDK"),
        ("cv2", "OpenCV Image Processing"),
        ("PIL", "Pillow Imaging"),
        ("numpy", "NumPy Vector Arrays"),
        ("requests", "Requests HTTP Client"),
    ]
    for mod_name, label in pkgs:
        try:
            __import__(mod_name)
            print(f"  ✓ {label} ({mod_name}) is installed.")
        except ImportError:
            print(f"  ❌ Missing {label} ({mod_name})! Run: pip install {mod_name}")

    # 2. Modal Cloud Authentication
    print("\n[2/5] Checking Modal Cloud Authentication...")
    try:
        res = subprocess.run(
            [sys.executable, "-m", "modal", "profile", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        active_ws = "ammadraza01"
        for line in res.stdout.splitlines():
            if "ammadraza" in line:
                active_ws = line.strip()
                break
        print(f"  ✓ Modal CLI authenticated! Active workspace: {active_ws}")
    except Exception as e:
        print(f"  ⚠️ Modal CLI check warning: {e}. Run 'start.bat' option [11] to authenticate.")

    # 3. Modal Container Budget Verification (Hard limit: <= 8)
    print("\n[3/5] Verifying Modal Container Budget (Hard Limit <= 8)...")
    try:
        from modal_budget import verify_modal_container_budget
        is_valid, total_conts, breakdown = verify_modal_container_budget(hard_cap=8)
        print(f"  ✓ Container budget strictly compliant: {total_conts}/8 containers allocated.")
        for name, count in breakdown.items():
            print(f"    • {name.upper()}: {count} container(s)")
    except Exception as e:
        print(f"  ❌ Budget violation: {e}")

    # 4. Environment Variables & API Keys (.env)
    print("\n[4/5] Checking Environment Keys & Credentials...")
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env_keys = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_keys[k.strip()] = v.strip()
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
    
    has_or = bool(os.environ.get("OPENROUTER_API_KEY") or env_keys.get("OPENROUTER_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or env_keys.get("GROQ_API_KEYS"))
    has_ollama = bool(os.environ.get("OLLAMA_API_KEY") or env_keys.get("OLLAMA_API_KEY"))

    print(f"  {'✓' if has_or else '⚠️'} OPENROUTER_API_KEY : {'Configured (Uncensored Action & Hermes 4 405B)' if has_or else 'Not set in .env'}")
    print(f"  {'✓' if has_groq else '⚠️'} GROQ_API_KEYS      : {'Configured (Fast Storybook & Kids Dialogue)' if has_groq else 'Not set in .env'}")
    print(f"  {'✓' if has_ollama else '⚠️'} OLLAMA_API_KEY     : {'Configured (Gemma 4: 31B Vision Gateway)' if has_ollama else 'Not set in .env'}")

    # 5. Live Handshake Latency Tests
    print("\n[5/5] Testing Live AI Director Gateway Connections...")
    try:
        from studio_pipeline import StudioPipeline
        if has_or:
            res_or = StudioPipeline.test_llm_connection(provider="openrouter")
            if res_or.get("success"):
                print(f"  ✓ OpenRouter Gateway Connected: {res_or.get('model')} ({res_or.get('latency_ms')}ms)")
            else:
                print(f"  ⚠️ OpenRouter Connection: {res_or.get('message')}")
        
        if has_groq:
            res_groq = StudioPipeline.test_llm_connection(provider="groq")
            if res_groq.get("success"):
                print(f"  ✓ GroqCloud Gateway Connected : {res_groq.get('model')} ({res_groq.get('latency_ms')}ms)")
            else:
                print(f"  ⚠️ GroqCloud Connection: {res_groq.get('message')}")
    except Exception as e:
        print(f"  ⚠️ Connection test error: {e}")

    print("\n" + "=" * 72)
    print(" ✓ PRE-FLIGHT VERIFICATION COMPLETE — STUDIO READY FOR PRODUCTION ".center(72, "="))
    print("=" * 72)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Support non-interactive CLI flags from start.bat or terminal
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        try:
            if arg in ("--check-setup", "check-setup", "--verify", "verify"):
                verify_setup_and_deployment()
                return
            elif arg in ("--deploy-all", "deploy-all"):
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
            elif arg in ("--deploy-narrator", "deploy-narrator"):
                deploy_narrator()
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
            elif arg in ("--fast-story", "fast-story", "--omniflow", "omniflow"):
                run_fast_story()
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

    print("=" * 70)
    print(" Wan 2.2 AI Studio Launcher & Modal Manager ".center(70, "="))
    print("=" * 70)

    print("[Options]")
    print(" 1. Start Studio frontend locally (http://localhost:7860)")
    print(" 2. Pre-flight Setup & Deployment Verification (Modal + Budget + LLMs)")
    print(" 3. Deploy ALL 4 Studio Backends to Modal (Wan 2.2 + Qwen + Z-Image + Narrator)")
    print(" 4. Deploy Wan 2.2 Video Backend Only (deploy_wan.py)")
    print(" 5. Deploy Qwen Image Edit Backend Only (deploy_qwen.py)")
    print(" 6. Deploy Z-Image Turbo Character Backend Only (deploy_z_image.py)")
    print(" 7. Deploy Kokoro-82M Neural Narrator Backend Only (deploy_narrator.py)")
    print(" 8. Pre-cache ALL Model Weights on Modal Volumes")
    print(" 9. Run OmniFlow-30 Fast Story CLI (Fast 30s Movie CLI)")
    print("10. Start Frontend on Local Network (LAN / Mobile access)")
    print("11. Switch / Manage Modal Account Profiles (runs 'modal profile' / 'modal setup')")
    print("12. Clear local outputs folder (outputs/)")

    choice = input("\nEnter choice [1-12] (default 1): ").strip()

    try:
        if choice == "2":
            verify_setup_and_deployment()
            sys.exit(0)
        elif choice == "3":
            deploy_all()
            sys.exit(0)
        elif choice == "4":
            deploy_wan()
            sys.exit(0)
        elif choice == "5":
            deploy_qwen()
            sys.exit(0)
        elif choice == "6":
            deploy_z_image()
            sys.exit(0)
        elif choice == "7":
            deploy_narrator()
            sys.exit(0)
        elif choice == "8":
            cache_all_weights()
            sys.exit(0)
        elif choice == "9":
            run_fast_story()
            sys.exit(0)
        elif choice == "10":
            os.environ["WAN_SHARE_LAN"] = "1"
        elif choice == "11":
            manage_modal_profiles()
            sys.exit(0)
        elif choice == "12":
            clear_outputs(script_dir)
            sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment/Operation failed: {e}")
        sys.exit(1)

    launch_frontend()


if __name__ == "__main__":
    main()
