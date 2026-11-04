import sys
import os

# Ensure the root of the project and internal folders are on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'brain'))
sys.path.insert(0, os.path.join(root_dir, 'data'))

# Import the configured Flask app
from app import app
