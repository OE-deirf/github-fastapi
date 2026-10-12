from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Churn serving API",
    version="v0.1",
    summary="FastApi Demo 1",
)


class Item(BaseModel):
    name: str
    price: float


@app.get("/world")
def world() -> dict[str, str]:
    return {'Hello': 'World'}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = "") -> dict[str, Any]:
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item) -> dict[str, Any]:
    return {"item_name": item.name, "item_id": item_id}
