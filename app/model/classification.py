import os
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Request(BaseModel):
    text: str


class PredictionEntity(BaseModel):
    label: str
    score: float


Response = List[PredictionEntity]


class MultilabelClassifier:
    """
    Class for Multi-Label Sequence Classification.

    Example
    --------
    >>> classifier = MultilabelClassifier(model_name_or_path)
    >>> json_output = classifier.predict(text)
    """

    def __init__(self, model_name_or_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        with open(model_name_or_path + os.sep + 'labels.txt', 'r') as f:
            self.labels = f.read().splitlines()
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def jsonify(response: Response) -> List[dict]:
        json_output = [prediction_entity.__dict__ for prediction_entity in response]
        return json_output

    def predict(self, request: Request) -> Response:
        """
        Runs predictions for the given text.
        """

        # apply model
        encoding = self.tokenizer(request.text, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        outputs = self.model(**encoding)

        # jsonify
        logits = outputs.logits
        probs = self.sigmoid(logits.squeeze().cpu())
        predictions = [
            PredictionEntity(label=self.id2label[idx], score=probs[idx].item())
            for idx, label in enumerate(probs)
        ]
        predictions = sorted(predictions, key=lambda prediction_entity: prediction_entity.score, reverse=True)
        response: Response = predictions

        # print positive predictions
        predictions = (probs >= 0.5).int()
        positive_predictions = [self.id2label[idx] for idx, label in enumerate(predictions) if label == 1]
        print("Positive predictions: ", ', '.join(positive_predictions))
        return response
