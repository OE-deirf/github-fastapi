from fastapi import FastAPI

app = FastAPI(
    title="Churn serving API",
    version="v0.1",
    summary="FastApi Demo 0",
)


@app.get("/world")
def world() -> dict[str, str]:
    return {'Hello': 'World'}

