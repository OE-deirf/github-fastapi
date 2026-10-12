from fastapi import FastAPI

app = FastAPI(
    title="Churn serving API",
    version="v0.1",
    summary="FastApi Demo 0",
)


@app.get("/world")
def world() -> dict[str, str]:
    return {'Hello': 'World'}


@app.get("/items/{item_id}")
def read_item(item_id: str, q: str = "") -> dict[str, str]:
    return {"item_id": item_id, "q": q}
