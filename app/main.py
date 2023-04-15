from fastapi import FastAPI
from app.model.classification import Request, Response
from app.model.modeling import classifier
from app.model.modeling import __version__ as model_version


app = FastAPI()


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=Response)
def predict(payload: Request) -> Response:
    response = classifier.predict(payload)
    return classifier.jsonify(response)
