from pydantic import BaseModel


class Urls(BaseModel):
    passing_url: str
    rushing_url: str
    receiving_url: str
