import os
import jwt

from datetime import datetime, timedelta, timezone
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from typing import Annotated, Any, AsyncGenerator
from fastapi import HTTPException, status
from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from pydantic import BaseModel


SECRET_KEY: str = os.environ.get("SECRET_KEY", "very-secret-key")
ALGORITHM = "HS512"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash: PasswordHash = PasswordHash.recommended()
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
        summary="FastApi Demo 4",
        lifespan=lifespan)


class UserDB():
    fake_users_db: dict[str, dict[str, Any]] = {
        "alice": {
            "username": "alice",
            "hashed_password": password_hash.hash("secret"),
            "active": True,
        },
    }

    def get_user(self, username: str) -> dict[str, Any]:
        user: dict[str, Any] = self.fake_users_db.get(username, {})
        return user

    def get_hashed_password(self, username: str) -> str:
        user: dict[str, Any] = self.fake_users_db.get(username, {})
        return user.get("hashed_password", "")

    def get_active(self, username: str) -> bool:
        user: dict[str, Any] = self.fake_users_db.get(username, {})
        return user.get("active", False)


class User(BaseModel):
    username: str
    active: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


def authenticate_user(username: str, password: str) -> str:
    active_user: bool = UserDB().get_active(username)
    if not active_user:
        return ""

    hashed_password: str = UserDB().get_hashed_password(username)
    try:
        pwd_validation: bool = password_hash.verify(password, hashed_password)
    except UnknownHashError:
        pwd_validation = False
    valid_username: str = UserDB().get_user(username).get("username", "")
    return valid_username if pwd_validation else ""


def create_access_token(username: str) -> str:
    expire: datetime = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    print(expire)
    return jwt.encode({"username": username, "expire": expire.time().isoformat()}, SECRET_KEY, algorithm=ALGORITHM)


@app.post("/token", response_model=Token, tags=["token"])
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    username: str = authenticate_user(form_data.username, form_data.password)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(username))


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: dict[str, Any] = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except InvalidTokenError:
        raise credentials_exception
    username: str = payload.get("username", "")
    active_user: bool = UserDB().get_active(username) if username else False
    if not active_user:
        raise credentials_exception
    user: dict[str, Any] = UserDB().get_user(username)
    return User(**user)


@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]) -> dict[str, str]:
    return {"token": token}


@app.get("/predict", dependencies=[Depends(oauth2_scheme)])
async def predict(tag: float) -> dict[str, Any]:
    result: float = ml_models["answer_to_everything"](tag)
    return {"result": result}
