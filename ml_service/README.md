# ML Service: Clickbait Checker (Multilingual)

This service uses a pre-trained multilingual transformer model to predict whether a video (YouTube, Dailymotion, etc.) is clickbait based on its metadata, thumbnail text, and transcript.

## Features
- Multilingual support (XLM-RoBERTa based)
- Input: title, description, tags, thumbnail text, transcript
- Output: Clickbait YES/NO and confidence score

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI server:**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## Usage

Send a POST request to `/predict` with JSON body:
```json
{
  "title": "...",
  "description": "...",
  "tags": ["tag1", "tag2"],
  "thumbnail_text": "...",
  "transcript": "..."
}
```

Response:
```json
{
  "clickbait": "YES",
  "confidence": 0.87
}
```

## Notes
- The current model is a sentiment classifier for demonstration. For production, fine-tune on a clickbait dataset.
- Supports all major languages covered by XLM-RoBERTa. 