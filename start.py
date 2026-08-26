import os
import sys
import subprocess
import webbrowser
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

def install_package(package_name, pip_name=None):
    pip_name = pip_name or package_name
    print(f"Installing {package_name} package locally...")
    subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)


def main():
    print("=" * 60)
    print(" Wan 2.2 Studio Localhost Runner ".center(60, "="))
    print("=" * 60)

    # Make sure we are in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("[Options]")
    print("1. Start Wan 2.2 Studio frontend locally")
    print("2. Deploy/Redeploy Wan 2.2 backend to Modal")
    print("3. Deploy/Redeploy Qwen Image Edit backend to Modal")
    print("4. Pre-cache Wan 2.2 model weights on Modal")
    print("5. Pre-cache Qwen Image Edit model weights on Modal")
    print("6. Start frontend on Local Network (mobile access)")
    print("7. Reconnect/Setup Modal account (runs 'modal setup')")
    print("8. Clear local outputs folder (outputs/)")

    choice = input("\nEnter choice [1/2/3/4/5/6/7/8] (default 1): ").strip()

    if choice == "6":
        os.environ["WAN_SHARE_LAN"] = "1"

    if choice == "7":
        print("\n---> Setting up Modal account...")
        try:
            import modal
        except ImportError:
            install_package("modal")
        
        cmd = [sys.executable, "-m", "modal", "setup"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Modal setup completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Modal setup failed: {e}")
            sys.exit(1)
        sys.exit(0)

    if choice == "2":
        print("\n---> Deploying Wan 2.2 backend to Modal...")
        try:
            import modal
        except ImportError:
            install_package("modal")

        cmd = [sys.executable, "-m", "modal", "deploy", "deploy_wan.py"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Wan 2.2 backend deployed to Modal!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Deployment failed: {e}")
            sys.exit(1)
        sys.exit(0)

    elif choice == "3":
        print("\n---> Deploying Qwen Image Edit backend to Modal...")
        try:
            import modal
        except ImportError:
            install_package("modal")

        cmd = [sys.executable, "-m", "modal", "deploy", "deploy_qwen.py"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Qwen Image Edit backend deployed to Modal!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Deployment failed: {e}")
            sys.exit(1)
        sys.exit(0)

    elif choice == "4":
        print("\n---> Caching Wan 2.2 model weights in Modal volume...")
        try:
            import modal
        except ImportError:
            install_package("modal")

        cmd = [sys.executable, "-m", "modal", "run", "deploy_wan.py::download_all_weights"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Wan 2.2 model weights cached successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Cache preloading failed: {e}")
            sys.exit(1)
        sys.exit(0)

    elif choice == "5":
        print("\n---> Caching Qwen Image Edit model weights in Modal volume...")
        try:
            import modal
        except ImportError:
            install_package("modal")

        cmd = [sys.executable, "-m", "modal", "run", "deploy_qwen.py::download_qwen_weights"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Qwen Image Edit model weights cached successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Cache preloading failed: {e}")
            sys.exit(1)
        sys.exit(0)

    elif choice == "8":
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
                    except Exception as e:
                        print(f"Failed to delete {name}: {e}")
            os.makedirs(outputs_dir, exist_ok=True)
            print(f"\n✓ Cleared {count} file(s) from outputs folder.")
        else:
            os.makedirs(outputs_dir, exist_ok=True)
            print("\n✓ Outputs folder was already empty.")
        sys.exit(0)

    # Start local app (Option 1 or Option 6)
    print("\n---> Checking local dependencies...")
    
    try:
        import gradio
    except ImportError:
        install_package("gradio", "gradio>=5.0,<7.0")

    try:
        import modal
    except ImportError:
        install_package("modal")

    try:
        import cv2
    except ImportError:
        install_package("opencv-python")

    try:
        from PIL import Image
    except ImportError:
        install_package("pillow")

    try:
        import numpy
    except ImportError:
        install_package("numpy")

    try:
        import requests
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
    
    # Run app.py locally
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

if __name__ == "__main__":
    main()
