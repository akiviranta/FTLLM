from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
from groq import Groq

model_name = "Pekkapuuma/distilbert-sentiment-classifier"
hf_token = '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'
Groq_key = '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'



# Initialize Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)

# Initialize Groq client
groq_client = Groq(api_key=Groq_key)

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request body
class SentimentRequest(BaseModel):
    text: str
    model: str

# Function to handle sentiment analysis with the Llama model through Groq
def analyze_llama_sentiment(text: str):
    llama_prompt = f"Classify the sentiment of this text as positive or negative: '{text}'"
    
    try:
        # Send request to Groq Llama model
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": llama_prompt}],
            model="llama-3.3-70b-versatile"  # Correct Llama model name
        )
        
        # Get sentiment from the response
        sentiment = chat_completion.choices[0].message.content.lower()
        if "positive" in sentiment:
            return "positive", 1.0
        elif "negative" in sentiment:
            return "negative", 1.0
        else:
            return "unknown", 0.0
        
    except Exception as e:
        print(f"Error with Llama API: {e}")
        return "error", 0.0

# API endpoint for sentiment analysis
@app.post("/analyze/")
async def analyze_sentiment(request: SentimentRequest):
    if request.model == "custom":
        # Hugging Face custom model sentiment analysis
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sentiment_score, sentiment_label = torch.max(probs, dim=-1)
        
        sentiment = "positive" if sentiment_label.item() == 1 else "negative"
        confidence = probs[0][sentiment_label.item()].item()
        
        return {"sentiment": sentiment, "confidence": confidence}
    
    elif request.model == "llama":
        # Groq Llama model sentiment analysis
        sentiment, confidence = analyze_llama_sentiment(request.text)
        return {"sentiment": sentiment, "confidence": confidence}
    
    else:
        return {"error": "Model not supported"}

# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)