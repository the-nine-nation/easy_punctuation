from typing import Union
import uvicorn
from fastapi import FastAPI
from Inferences import gogo
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
@app.get("/yyy/{txt2txt}")
def read_root(txt2txt:str):
    print(txt2txt)
    return gogo(source_text=txt2txt)

