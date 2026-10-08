import sys
from pathlib import Path

# Ensure project root is on Python's path so pipeline and src imports resolve
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,          # keeps the interactive debugger and tracebacks
        use_reloader=False,  # one process, one model, no mid-request restarts
        threaded=True,       # static files still load while a query runs
    )