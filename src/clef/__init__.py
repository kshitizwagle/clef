import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "clef.app:app",
        host=os.environ.get("CLEF_HOST", "127.0.0.1"),
        port=int(os.environ.get("CLEF_PORT", "8000")),
    )
