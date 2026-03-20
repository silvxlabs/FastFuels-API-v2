import lib.config  # noqa: F401  Ensure .env is loaded before GCP client init
from api.app import app  # noqa: F401

if __name__ == "__main__":
    from uvicorn import run

    run("api.main:app", workers=1, reload=True, port=8080)
