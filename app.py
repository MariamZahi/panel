import subprocess

def run_streamlit_app():
    try:
        subprocess.run(["streamlit", "hello", "--server.address", "0.0.0.0", "--server.port", "8000"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)

if __name__ == "__main__":
    run_streamlit_app()
