import subprocess

def run_streamlit_app():
    try:
        subprocess.run(["streamlit", "run", "--server.port", "8000", "app_ui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)

if __name__ == "__main__":
    run_streamlit_app()
