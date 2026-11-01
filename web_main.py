import os
import sys

# Ensure root directory is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.web.app import run_web_app

if __name__ == '__main__':
    run_web_app(host="0.0.0.0", port=5000)
