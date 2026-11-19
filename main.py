from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import os
import tempfile

app = FastAPI()
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(400, "Only WAV and MP3 files are supported")

    path = None

    try:
        suffix = os.path.splitext(file.filename)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            path = temp.name
            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        segments, info = model.transcribe(path)

        result = []

        for segment in segments:
            result.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        return {
            "language": info.language,
            "text": " ".join(item["text"] for item in result),
            "segments": result
        }

    finally:
        if path and os.path.exists(path):
            os.remove(path)