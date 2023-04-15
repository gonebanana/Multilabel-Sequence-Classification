from app.model.classification import MultilabelClassifier, Request
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parents[1]

classifier = MultilabelClassifier(str(BASE_DIR / f"bert-finetuned-sem_eval-english-{__version__}"))


if __name__ == '__main__':
    text = "I'm happy I can finally train a model for multi-label classification"
    response = classifier.predict(Request(text=text))
    print(classifier.jsonify(response))
