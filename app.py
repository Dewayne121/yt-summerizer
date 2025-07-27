import os
import re
import logging
import subprocess
import tempfile
import glob
import json
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configure Google AI API ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("CRITICAL: GOOGLE_API_KEY environment variable not found.")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logger.info("Google AI SDK configured successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to configure Google AI SDK: {e}", exc_info=True)
    model = None

# --- FastAPI App Initialization ---
app = FastAPI(title="YouTube Summarizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class SummarizeRequest(BaseModel):
    youtube_url: str

# --- Core Logic Functions ---
def get_video_id(url: str):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None

def parse_vtt_to_structured_transcript(vtt_content: str):
    """
    Parses VTT content into a list of dictionaries with start time and text.
    Also returns the full plain text for summarization.
    """
    lines = vtt_content.strip().split('\n')
    structured_transcript = []
    full_text_lines = []
    current_text = ""
    start_time = ""

    for i, line in enumerate(lines):
        if '-->' in line:
            try:
                # Capture the start time from the timestamp line
                start_time = line.split('-->')[0].strip().split('.')[0] # Get time before milliseconds
                # The actual text is on the next line(s)
                text_parts = []
                j = i + 1
                while j < len(lines) and lines[j].strip() != '':
                    # Remove HTML-like tags from captions
                    cleaned_line = re.sub(r'<[^>]+>', '', lines[j])
                    text_parts.append(cleaned_line.strip())
                    j += 1
                
                current_text = " ".join(text_parts)
                
                if start_time and current_text:
                    # Avoid adding duplicate entries from multi-line captions
                    if not structured_transcript or structured_transcript[-1]['text'] != current_text:
                        structured_transcript.append({"time": start_time, "text": current_text})
                        full_text_lines.append(current_text)

            except IndexError:
                continue # Skip malformed timestamp lines
                
    plain_text = " ".join(full_text_lines)
    return structured_transcript, plain_text


def get_transcript_with_ytdlp(youtube_url: str):
    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format provided.")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            cmd = ['yt-dlp', '--write-auto-subs', '--write-subs', '--sub-langs', 'en.*', '--sub-format', 'vtt', '--skip-download', '--output', f'{temp_dir}/%(id)s.%(ext)s', youtube_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True, encoding='utf-8')
            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            if not vtt_files:
                raise RuntimeError("No English subtitles were found for this video.")
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            return parse_vtt_to_structured_transcript(vtt_content)
        except subprocess.TimeoutExpired:
            raise RuntimeError("The transcript download timed out.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Could not fetch subtitles: {e.stderr}")

def summarize_with_google_ai(transcript: str, word_count: int):
    if not model:
        raise RuntimeError("AI model is not available due to a configuration error.")

    prompt = f"""
    As an expert analyst, provide a comprehensive summary of the following video transcript.
    Format your output in Markdown with two distinct sections:
    
    ## Quick Summary
    A concise paragraph capturing the main point and conclusion.
    
    ## Key Takeaways
    A bulleted list of the 4-6 most important points.
    
    ---
    Transcript: "{transcript}"
    """
    try:
        response = model.generate_content(prompt)
        summary_markdown = response.text + f"\n\n*AI analysis powered by Google Gemini. Processed approximately {word_count:,} words.*"
        return summary_markdown
    except Exception as e:
        raise RuntimeError(f"The AI summarization failed. Error: {e}")

@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    try:
        structured_transcript, plain_text = get_transcript_with_ytdlp(request.youtube_url)
        word_count = len(plain_text.split())
        
        if word_count < 50:
             return JSONResponse(
                content={
                    "summary": "This video is too short to summarize.", 
                    "transcript": structured_transcript
                }
            )

        summary_markdown = summarize_with_google_ai(plain_text, word_count)
        
        return JSONResponse(
            content={
                "summary": summary_markdown, 
                "transcript": structured_transcript
            }
        )
    except (ValueError, RuntimeError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"An unexpected internal error occurred: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})
