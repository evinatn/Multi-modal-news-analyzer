# inference.py - NEW SEPARATE FILE
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import sys

class NewsClassifier:
    def __init__(self, model_path="./saved_news_classifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
    
    def predict(self, text):
        encoding = self.tokenizer(text, return_tensors='pt', 
                                 truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        return self.label_encoder.inverse_transform([prediction])[0]

if __name__ == "__main__":
    classifier = NewsClassifier()
    text = sys.argv[1] if len(sys.argv) > 1 else "Sample news text"
    result = classifier.predict(text)
    print(f"Predicted category: {result}")