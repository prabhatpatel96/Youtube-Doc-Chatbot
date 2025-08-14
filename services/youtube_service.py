import os
import yt_dlp
import webvtt

def _download_subs(video_url: str, lang: str = "en") -> str:
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "skip_download": True,
        "outtmpl": "%(id)s.%(ext)s",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=True)
        vid = result.get("id")
        path = f"{vid}.{lang}.vtt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subtitle file not found: {path}")
        return path

def _vtt_to_text(vtt_path: str) -> str:
    transcript = []
    for caption in webvtt.read(vtt_path):
        text = caption.text.strip()
        if text:
            transcript.append(text)
    return " ".join(transcript).strip()

def fetch_youtube_transcript(url: str, lang: str = "en") -> str:
    try:
        vtt = _download_subs(url, lang=lang)
        text = _vtt_to_text(vtt)
        try:
            os.remove(vtt)
        except Exception:
            pass
        return text
    except Exception as e:
        return f"[ERROR] {e}"
