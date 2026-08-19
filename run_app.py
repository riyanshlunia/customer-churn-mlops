import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def print_help():
    print("================ Enterprise Churn Prediction Platform ================")
    print("Usage: python run_app.py [command]")
    print("\nAvailable Commands:")
    print("  train     - Run model training (LightGBM & XGBoost on Telco & Bank datasets)")
    print("  api       - Launch FastAPI microservice on http://127.0.0.1:8000")
    print("  dashboard - Launch Streamlit interactive dashboard on http://localhost:8501")
    print("  all       - Train models & print startup instructions")
    print("=======================================================================")

def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "train":
        print("Training models...")
        subprocess.run([sys.executable, "src/model_training.py"], cwd=BASE_DIR, check=True)
    elif cmd == "api":
        print("Launching FastAPI REST Service on http://127.0.0.1:8000 ...")
        subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"], cwd=BASE_DIR, check=True)
    elif cmd == "dashboard":
        print("Launching Streamlit Dashboard on http://localhost:8501 ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], cwd=BASE_DIR, check=True)
    elif cmd == "all":
        print("Step 1: Training models...")
        subprocess.run([sys.executable, "src/model_training.py"], cwd=BASE_DIR, check=True)
        print("\nModels trained successfully!")
        print("To launch the Streamlit dashboard, run:")
        print("  python run_app.py dashboard")
        print("To launch the FastAPI service, run:")
        print("  python run_app.py api")
    else:
        print(f"Unknown command: {cmd}")
        print_help()

if __name__ == "__main__":
    main()
