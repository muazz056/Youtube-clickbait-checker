import os
# Prepend ffmpeg bin directory to PATH so subprocess can find it
ffmpeg_dir = r"C:/Users/X13 YOGA/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.1.1-full_build/bin"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_dir, "ffmpeg.exe")
from fastapi import FastAPI, File, UploadFile
import whisper
import tempfile
import uuid

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        print("Received file:", file.filename)
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = ".mp3"  # fallback
        temp_path = os.path.join(tempfile.gettempdir(), f"whisper_{uuid.uuid4().hex}{ext}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        print("Saved temp file:", temp_path)
        print('Current working directory:', os.getcwd())
        print('Temp file exists:', os.path.exists(temp_path))
        print('Temp file path:', temp_path)
        print("FFMPEG_BINARY:", os.environ.get("FFMPEG_BINARY"))
        print("PATH:", os.environ.get("PATH"))
        result = model.transcribe(temp_path)
        os.remove(temp_path)
        print("Transcription result:", result["text"])
        return {"transcript": result["text"]}
    except Exception as e:
        print("Whisper transcription error:", e)
        return {"error": str(e)}

model = whisper.load_model("base")
# result = model.transcribe(r"D:\Stream_clickbaitChecker\backend\audio_1751039932481.mp3")
# print(result["text"]) 