import os
ffmpeg_dir = r"C:/Users/X13 YOGA/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.1.1-full_build/bin"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_dir, "ffmpeg.exe")

# Now import everything else
from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BlipProcessor, BlipForConditionalGeneration
import torch
import uvicorn
import numpy as np
import whisper
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import re
import requests
import json
import subprocess
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import uuid
import io
import base64
import openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load English-only model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# For demonstration, we use a sentiment model; in production, fine-tune on clickbait data

class VideoInput(BaseModel):
    title: str
    description: str
    tags: list[str]
    thumbnail_text: str
    transcript: str

# Helper to translate text to English
def translate_to_english(text):
    if not text.strip():
        return text
    try:
        prompt = f"Translate the following text to English. If it is already in English, return it unchanged.\n\nText: {text}"
        result = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text to English."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512
        )
        translation = result.choices[0].message.content.strip()
        return translation
    except Exception as e:
        print("OpenAI translation error:", e)
        return text  # fallback to original if translation fails

def text_similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    vect = TfidfVectorizer().fit([a, b])
    tfidf = vect.transform([a, b])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(sim)

def tags_in_transcript(tags, transcript):
    transcript_lower = transcript.lower()
    return any(tag.lower() in transcript_lower for tag in tags)

@app.post("/predict")
async def predict_clickbait(data: VideoInput):
    # Translate all fields to English
    title_en = translate_to_english(data.title)
    description_en = translate_to_english(data.description)
    tags_en = [translate_to_english(tag) for tag in data.tags]
    thumbnail_text_en = translate_to_english(data.thumbnail_text)
    transcript_en = translate_to_english(data.transcript)

    # Compute similarities
    title_sim = text_similarity(title_en, transcript_en)
    desc_sim = text_similarity(description_en, transcript_en)
    thumb_sim = text_similarity(thumbnail_text_en, transcript_en)
    tags_match = tags_in_transcript(tags_en, transcript_en)

    # Concatenate all text fields for clickbait model
    text = f"Title: {title_en}\nDescription: {description_en}\nTags: {' '.join(tags_en)}\nThumbnail: {thumbnail_text_en}\nTranscript: {transcript_en}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    clickbait_score = float(probs[1])
    is_clickbait = clickbait_score > 0.5
    verdict = "YES" if is_clickbait else "NO"

    # Generate summary
    summary = f"Title/Transcript similarity: {title_sim:.2f}. Thumbnail/Transcript similarity: {thumb_sim:.2f}. Description/Transcript similarity: {desc_sim:.2f}. Tags match: {tags_match}. Verdict: {verdict}."
    if is_clickbait:
        summary += " This video is likely clickbait."
    else:
        summary += " This video is likely NOT clickbait."

    return {
        "title_transcript_similarity": title_sim,
        "tags_match": tags_match,
        "thumbnail_text_match": thumb_sim,
        "description_match": desc_sim,
        "verdict": verdict,
        "clickbait_report_summary": summary,
        "clickbait": verdict,
        "confidence": clickbait_score
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        result = model.transcribe(tmp.name)
    return {"transcript": result["text"]}

# Helper to extract video ID from URL

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

# Helper to get YouTube video metadata using yt-dlp

def get_video_metadata_yt_dlp(video_url):
    try:
        # Run yt-dlp as a subprocess to get JSON metadata
        result = subprocess.run([
            'yt-dlp',
            '-j',
            video_url
        ], capture_output=True, text=True, check=True)
        meta = json.loads(result.stdout)
        return meta
    except Exception as e:
        print('yt-dlp error:', e)
        return {}

# Helper to download and OCR thumbnail

reader = easyocr.Reader(['en'])

def extract_thumbnail_text(thumbnail_url):
    try:
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            with open('temp_thumbnail.jpg', 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            result = reader.readtext('temp_thumbnail.jpg', detail=0)
            os.remove('temp_thumbnail.jpg')
            return ' '.join(result)
    except Exception as e:
        print('Thumbnail OCR error:', e)
    return ''

# Load Whisper model once at startup
whisper_model = whisper.load_model("base")

# Set the path to yt-dlp.exe (update this path if needed)
YT_DLP_PATH = r"yt-dlp"  # If yt-dlp is in PATH, this works. Otherwise, set full path, e.g. r"C:\Users\YourUser\AppData\Local\Programs\Python\Python310\Scripts\yt-dlp.exe"

def transcribe_with_whisper(video_url):
    temp_audio = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.mp3")
    try:
        print("[DEBUG] PATH:", os.environ.get("PATH"))
        print("[DEBUG] FFMPEG_BINARY:", os.environ.get("FFMPEG_BINARY"))
        print("[DEBUG] YT_DLP_PATH:", YT_DLP_PATH)
        # Download best audio only
        subprocess.run([
            YT_DLP_PATH, "-f", "bestaudio", "-o", temp_audio, video_url
        ], check=True)
        print("[DEBUG] Temp audio exists after yt-dlp:", os.path.exists(temp_audio), temp_audio)
        # Transcribe with Whisper
        result = whisper_model.transcribe(temp_audio)
        transcript_text = result["text"]
        detected_lang = result.get("language", "en")
        print(f"Whisper detected language: {detected_lang}")
        # If not English, translate to English
        if detected_lang != "en":
            transcript_text = translate_to_english(transcript_text)
        return transcript_text
    except Exception as e:
        print("Whisper transcription error:", e)
        return ""
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

# Load BLIP image captioning model once at startup
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_scene_description_openai(thumbnail_url):
    try:
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            image_bytes = response.content
            prompt = "Describe the scene in this YouTube thumbnail in detail, including any visual elements, actions, and context, but do not just repeat the text."
            result = openai.chat.completions.create(
                model="gpt-4-1106-vision-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that describes images."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode('utf-8')}}
                    ]}
                ],
                max_tokens=512
            )
            scene_description = result.choices[0].message.content.strip()
            return scene_description
    except Exception as e:
        print("OpenAI Vision scene extraction error:", e)
    return ""

def extract_scene_description(thumbnail_url):
    # Try OpenAI Vision first
    openai_desc = extract_scene_description_openai(thumbnail_url)
    if openai_desc:
        return openai_desc
    # Fallback to previous logic (OCR+BLIP)
    ocr_text = extract_thumbnail_text(thumbnail_url)
    if ocr_text and len(ocr_text.split()) > 3:
        return ocr_text
    try:
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            inputs = blip_processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
    except Exception as e:
        print("Scene extraction error:", e)
    return ocr_text or ""

@app.post("/extract_video_data")
async def extract_video_data(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return {"success": False, "error": "No URL provided"}
    video_id = extract_video_id(url)
    # Get metadata from yt-dlp
    meta = get_video_metadata_yt_dlp(url)
    # Extract all language fields if available
    def extract_multilang(field):
        # If yt-dlp provides a dict of languages, extract all
        val = meta.get(field, '')
        if isinstance(val, dict):
            return {lang: val[lang] for lang in val}
        return {'default': val}
    # Title, description, tags (if multilingual)
    titles = extract_multilang('title')
    descriptions = extract_multilang('description')
    tags = extract_multilang('tags')
    # Thumbnail
    thumbnail_url = meta.get('thumbnail', '')
    # OCR on thumbnail (only one language, unless you have multiple thumbnails)
    thumbnail_text = extract_thumbnail_text(thumbnail_url) if thumbnail_url else ''
    # Scene description from thumbnail
    scene_description = extract_scene_description(thumbnail_url) if thumbnail_url else ''
    # Get transcript in all available languages
    transcripts = {}
    transcripts_en = {}
    transcript_source = None
    # Helper to extract text from transcript segment
    def get_text(seg):
        if isinstance(seg, dict):
            return seg.get('text', '')
        return getattr(seg, 'text', '')

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print("Available transcript languages:")
        # Try manual transcripts first
        for t in transcript_list._manually_created_transcripts.values():
            print(f" - {t.language_code}: {t.language} (manual)")
        for t in transcript_list._generated_transcripts.values():
            print(f" - {t.language_code}: {t.language} (auto)")
        # Fetch all available transcripts in their original languages
        found = False
        # Manual transcripts
        for transcript in transcript_list._manually_created_transcripts.values():
            lang = transcript.language_code
            try:
                segs = transcript.fetch()
                text = ' '.join([get_text(seg) for seg in segs])
                transcripts[lang] = text
                if lang == 'en':
                    transcripts_en[lang] = text
                else:
                    try:
                        translated = transcript.translate('en')
                        segs_en = translated.fetch()
                        text_en = ' '.join([get_text(seg) for seg in segs_en])
                        transcripts_en[lang] = text_en
                    except Exception as e:
                        print(f"Translation to English failed for {lang}: {e}")
                        transcripts_en[lang] = translate_to_english(text)
                if not transcript_source:
                    transcript_source = 'manual'
                found = True
            except Exception as e:
                print(f"Transcript fetch failed for {lang}: {e}")
                continue
        # Auto-generated transcripts
        for transcript in transcript_list._generated_transcripts.values():
            lang = transcript.language_code
            try:
                segs = transcript.fetch()
                text = ' '.join([get_text(seg) for seg in segs])
                transcripts[lang] = text
                if lang == 'en':
                    transcripts_en[lang] = text
                else:
                    try:
                        translated = transcript.translate('en')
                        segs_en = translated.fetch()
                        text_en = ' '.join([get_text(seg) for seg in segs_en])
                        transcripts_en[lang] = text_en
                    except Exception as e:
                        print(f"Translation to English failed for {lang}: {e}")
                        transcripts_en[lang] = translate_to_english(text)
                if not transcript_source:
                    transcript_source = 'auto'
                found = True
            except Exception as e:
                print(f"Transcript fetch failed for {lang}: {e}")
                continue
    except Exception as e:
        print('Transcript extraction error:', e)
    # Translate all fields to English
    def translate_dict(d):
        return {k: translate_to_english(v) for k, v in d.items() if v}
    titles_en = translate_dict(titles)
    descriptions_en = translate_dict(descriptions)
    tags_en = {k: [translate_to_english(tag) for tag in v] if isinstance(v, list) else [translate_to_english(v)] for k, v in tags.items() if v}
    thumbnail_text_en = translate_to_english(thumbnail_text)
    # After transcript extraction logic
    # If no transcript found from YouTube, use Whisper
    if not transcripts_en:
        print("No transcript found from YouTube, using Whisper...")
        whisper_transcript = transcribe_with_whisper(url)
        if whisper_transcript:
            transcripts_en["whisper"] = whisper_transcript
            transcript_source = 'whisper'
    # For transcript, concatenate all English versions
    transcript_en = ' '.join(transcripts_en.values())
    # Concatenate all English versions for each field
    title_en = ' '.join(titles_en.values())
    description_en = ' '.join(descriptions_en.values())
    tags_en_flat = [tag for tagslist in tags_en.values() for tag in tagslist]
    return {
        "success": True,
        "title": title_en,
        "description": description_en,
        "tags": tags_en_flat,
        "thumbnail": thumbnail_url,
        "thumbnail_text": thumbnail_text_en,
        "scene_description": scene_description,
        "transcript": transcript_en,
        "transcript_source": transcript_source
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 