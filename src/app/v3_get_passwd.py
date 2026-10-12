from typing import Annotated, Any, AsyncGenerator
from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
ml_models: dict[str, Any] = {}


def fake_answer_to_everything_ml_model(tag: float) -> float:
    return tag * 42


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield
    ml_models.clear()


app = FastAPI(
        title="Churn serving API",
        version="v0.1",
        summary="FastApi Demo 3",
        lifespan=lifespan)


@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> dict[str, Any]:
    return {"access_token": form_data.username, "token_type": "bearer"}


@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]) -> dict[str, str]:
    return {"token": token}


@app.get("/predict")
async def predict(tag: float) -> dict[str, Any]:
    result: float = ml_models["answer_to_everything"](tag)
    return {"result": result}
