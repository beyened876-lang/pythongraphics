import os
import subprocess
import time
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(SCRIPT_DIR, 'fake_news_detection', 'app_fixed.py')

if __name__ == '__main__':
    print('Starting Streamlit app...')
    process = subprocess.Popen([
        'streamlit',
        'run',
        APP_PATH,
        '--server.headless',
        'false',
    ], cwd=SCRIPT_DIR)

    time.sleep(5)
    webbrowser.open('http://localhost:8501')
    print('The app should open in your default browser shortly.')
    process.wait()
