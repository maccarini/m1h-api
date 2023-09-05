from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    m1d_api_token: str
    openai_api_key: str


settings = Settings()