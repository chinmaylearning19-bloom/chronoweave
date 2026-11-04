"""
run.py — ChronoWeave Application Entry Point
Physics-Informed Neurosymbolic Supply Chain Simulator & Bullwhip Mitigation Engine
Domain: Indian Omnichannel Apparel Brand (Online D2C + 20 Retail Stores)
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, 'api'))
sys.path.insert(0, os.path.join(_here, 'brain'))
sys.path.insert(0, os.path.join(_here, 'data'))

from app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5051))
    print("=" * 70)
    print("  ChronoWeave: Neurosymbolic Omnichannel Supply Chain Simulator")
    print(f"  Target: Indian Apparel Brand (20 Stores + Online D2C)")
    print(f"  Dashboard: http://localhost:{port}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=True)
