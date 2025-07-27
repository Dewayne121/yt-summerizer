import os
import re
import logging
import subprocess
import tempfile
import glob
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configure the Google AI API ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using a fast and capable model from the Gemini family
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logger.info("Google AI SDK configured successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to configure Google AI SDK: {e}", exc_info=True)
    model = None # Ensure model is None if configuration fails

# --- FastAPI App Initialization ---
app = FastAPI(
    title="YouTube Video Summarizer API",
    description="A high-speed API for summarizing YouTube videos using Google's Gemini."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class SummarizeRequest(BaseModel):
    youtube_url: str

# --- Core Logic Functions ---
def get_video_id(url: str):
    if not url: return None
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match: return match.group(1)
    return None

def parse_vtt_content(vtt_content):
    lines = vtt_content.split('\n')
    text_lines = []
    for line in lines:
        line = line.strip()
        if (line.startswith('WEBVTT') or '-->' in line or not line or line.isdigit()):
            continue
        line = re.sub(r'<[^>]+>', '', line)
        if line:
            text_lines.append(line)
    return ' '.join(text_lines)

def get_transcript_with_ytdlp(youtube_url: str):
    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format provided.")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            cmd = [
                'yt-dlp', '--write-auto-subs', '--write-subs', '--sub-langs', 'en.*',
                '--sub-format', 'vtt', '--skip-download',
                '--output', f'{temp_dir}/%(id)s.%(ext)s', youtube_url
            ]
            logger.info(f"Running yt-dlp command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True, encoding='utf-8')
            
            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            if not vtt_files:
                raise RuntimeError("No English subtitles were found for this video.")
            
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                vtt_content = f.read()

            transcript_text = parse_vtt_content(vtt_content)
            if not transcript_text or len(transcript_text.split()) < 50:
                raise RuntimeError("The extracted transcript is too short to be summarized.")
            
            # Truncate very long transcripts to fit within API limits and keep costs low
            max_words = 15000
            words = transcript_text.split()
            if len(words) > max_words:
                transcript_text = " ".join(words[:max_words])
                logger.warning(f"Transcript truncated to {max_words} words.")

            return transcript_text

        except subprocess.TimeoutExpired:
            raise RuntimeError("The transcript download timed out. The video might be too long or the connection slow.")
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp failed: {e.stderr}")
            raise RuntimeError(f"Could not fetch subtitles. The video may not have them or they are in an unsupported format.")

def summarize_with_google_ai(transcript: str, word_count: int):
    """
    Generates a summary using the Google Gemini API.
    """
    if not model:
        raise RuntimeError("The AI model is not available due to a configuration error. Check the server logs.")

    # A more detailed prompt for higher quality summaries
    prompt = f"""
    As an expert analyst, your task is to provide a comprehensive summary of the following video transcript.

    Format your output in Markdown with two distinct sections:

    ## ⚡ Quick Summary
    A single, concise paragraph that captures the main point and conclusion of the video.

    ## 🔑 Key Takeaways
    A bulleted list of the 4-6 most important points, findings, or arguments from the transcript. Each point should be clear and easy to understand.

    ---

    **Transcript to Summarize:**
    "{transcript}"
    """

    try:
        logger.info("Sending request to Google Gemini API...")
        response = model.generate_content(prompt)
        # Add a footer to the response
        summary_markdown = response.text + f"\n\n*📊 AI analysis powered by Google Gemini. Processed approximately {word_count:,} words.*"
        return summary_markdown
    except Exception as e:
        logger.error(f"Google Gemini API call failed: {e}", exc_info=True)
        raise RuntimeError(f"The AI summarization failed. The service may be temporarily unavailable. Error: {e}")

# --- API Endpoint ---
@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    """
    The main API endpoint that receives a URL, gets the transcript, and returns an AI summary.
    """
    try:
        logger.info(f"Processing request for URL: {request.youtube_url}")
        
        # Step 1: Get transcript (fast)
        transcript = get_transcript_with_ytdlp(request.youtube_url)
        word_count = len(transcript.split())
        logger.info(f"Transcript fetched successfully with {word_count} words.")
        
        # Step 2: Generate summary with Google AI (very fast)
        summary_markdown = summarize_with_google_ai(transcript, word_count)
        logger.info("Summary generated successfully.")
        
        return JSONResponse(content={"summary": summary_markdown})
    
    except (ValueError, RuntimeError) as e:
        # User-facing errors (e.g., bad URL, no subtitles)
        logger.warning(f"Handled error for user: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        # Server-side errors
        logger.error(f"An unexpected internal error occurred: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred. Please try again later."})
