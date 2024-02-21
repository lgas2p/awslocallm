import uvicorn
from uvicorn.config import LOGGING_CONFIG

if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = (
        "%(asctime)s " + LOGGING_CONFIG["formatters"]["access"]["fmt"]
    )
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)