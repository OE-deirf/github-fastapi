from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from fastapi import FastAPI


def fake_answer_to_everything_ml_model(tag: float) -> float:
    return tag * 42


ml_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield
    ml_models.clear()


app = FastAPI(
        title="Churn serving API",
        version="v0.1",
        summary="FastApi Demo 2",
        lifespan=lifespan)


@app.get("/predict")
async def predict(tag: float) -> dict[str, Any]:
    result: float = ml_models["answer_to_everything"](tag)
    return {"result": result}
