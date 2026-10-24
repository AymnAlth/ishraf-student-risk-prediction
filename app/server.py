"""Server entrypoint for Ishraf Student Risk Early Warning System."""

import uvicorn
from src.core.config import HOST, PORT

def run():
    uvicorn.run("app.api:app", host=HOST, port=PORT, reload=False)

if __name__ == "__main__":
    run()
