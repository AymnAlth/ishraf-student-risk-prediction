"""Server entrypoint for Ishraf Student Risk Early Warning System."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from src.core.config import HOST, PORT

def run():
    uvicorn.run("app.api:app", host=HOST, port=PORT, reload=False)

if __name__ == "__main__":
    run()
