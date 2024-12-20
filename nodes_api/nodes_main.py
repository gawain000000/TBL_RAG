import os
import subprocess
import signal
import sys

# List to keep track of running processes
processes = []
conda_env_path = os.getenv('CONDA_DEFAULT_ENV')
conda_env_python_path = os.path.join(conda_env_path, "bin/python") if conda_env_path else sys.executable


def run_main_scripts():
    # Walk through the directory tree starting from the 'nodes_api' directory
    for root, dirs, files in os.walk('nodes_api'):
        # Check if 'main.py' exists in the current directory
        if 'main.py' in files:
            try:
                print(f"Running main.py in {root} using uvicorn...")
                # Run uvicorn using the Conda environment Python executable
                proc = subprocess.Popen(
                    [conda_env_python_path, "-m", "uvicorn", "main:app", "--reload"],
                    cwd=root  # Set the working directory to the directory containing main.py
                )
                processes.append(proc)
            except Exception as e:
                print(f"Failed to run main.py in {root}: {e}")


def stop_all_processes(signum, frame):
    print("\nStopping all processes...")
    # Terminate all running processes
    for proc in processes:
        if proc.poll() is None:  # Check if the process is still running
            proc.terminate()  # Terminate the process
            try:
                proc.wait(timeout=5)  # Give it time to exit gracefully
            except subprocess.TimeoutExpired:
                proc.kill()  # Force kill if it doesn't terminate in time
    sys.exit(0)


if __name__ == "__main__":
    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, stop_all_processes)
    signal.signal(signal.SIGTERM, stop_all_processes)

    # Run all main.py scripts
    run_main_scripts()

    # Keep the main script running to allow child processes to keep running
    try:
        # Loop indefinitely until a signal is caught
        while True:
            pass
    except KeyboardInterrupt:
        stop_all_processes(None, None)
