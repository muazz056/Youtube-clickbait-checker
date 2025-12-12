import sys
import re
import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from fastapi.middleware.cors import CORSMiddleware
from googletrans import Translator as GoogleTranslator
import easyocr
import requests
import os

def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)',
        r'youtu\.be/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url  # fallback: assume input is already a video ID

def extract_thumbnail_text(thumbnail_url):
    try:
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            with open('temp_thumbnail.jpg', 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            reader = easyocr.Reader(['en'])
            result = reader.readtext('temp_thumbnail.jpg', detail=0)
            os.remove('temp_thumbnail.jpg')
            return ' '.join(result)
    except Exception as e:
        print('Thumbnail OCR error:', e)
    return ''

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No URL or video ID provided"}))
        sys.exit(1)
    url = sys.argv[1]
    video_id = extract_video_id(url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = ' '.join([seg['text'] for seg in transcript])
        print(json.dumps({"success": True, "transcript": text}))
    except NoTranscriptFound:
        print(json.dumps({"success": False, "error": "No transcript found"}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    main() 