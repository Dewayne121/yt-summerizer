import os
import re
import logging
import subprocess
import tempfile
import glob
import json
import time
import random
from typing import Optional, Tuple
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

def convert_json_to_netscape(json_cookies_str: str) -> str:
    try:
        cookies = json.loads(json_cookies_str)
    except json.JSONDecodeError:
        return json_cookies_str
    netscape_lines = ["# Netscape HTTP Cookie File"]
    for cookie in cookies:
        domain = cookie.get("domain", "")
        if not domain: continue
        include_subdomains = "TRUE" if domain.startswith('.') else "FALSE"
        path = cookie.get("path", "/")
        secure = "TRUE" if cookie.get("secure", False) else "FALSE"
        expires = str(int(cookie.get("expirationDate", 0)))
        name = cookie.get("name", "")
        value = cookie.get("value", "")
        if name:
            netscape_lines.append("\t".join([domain, include_subdomains, path, secure, expires, name, value]))
    return "\n".join(netscape_lines)

def get_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match: return match.group(1)
    return None

def parse_vtt_to_structured_transcript(vtt_content: str) -> Tuple[list, str]:
    lines = vtt_content.strip().split('\n')
    structured_transcript = []
    full_text_lines = []
    for i, line in enumerate(lines):
        if '-->' in line:
            try:
                start_time = line.split('-->')[0].strip().split('.')[0]
                text_parts = []
                j = i + 1
                while j < len(lines) and lines[j].strip() != '':
                    cleaned_line = re.sub(r'<[^>]+>', '', lines[j])
                    text_parts.append(cleaned_line.strip())
                    j += 1
                current_text = " ".join(text_parts)
                if start_time and current_text and (not structured_transcript or structured_transcript[-1]['text'] != current_text):
                    structured_transcript.append({"time": start_time, "text": current_text})
                    full_text_lines.append(current_text)
            except IndexError: continue
    plain_text = " ".join(full_text_lines)
    return structured_transcript, plain_text

def get_random_user_agent() -> str:
    """Return a random user agent to help avoid rate limiting"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]
    return random.choice(user_agents)

def get_transcript_with_ytdlp(youtube_url: str, retry_count: int = 0) -> Tuple[list, str]:
    if not get_video_id(youtube_url): 
        raise ValueError("Invalid YouTube URL format provided.")

    proxy_url = os.getenv('PROXY_URL')
    cookies_data = os.getenv('YOUTUBE_COOKIES')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cookies_file_path = None
        if cookies_data:
            netscape_formatted_cookies = convert_json_to_netscape(cookies_data)
            cookies_file_path = os.path.join(temp_dir, 'cookies.txt')
            with open(cookies_file_path, 'w', encoding='utf-8') as f:
                f.write(netscape_formatted_cookies)
        
        try:
            # Add random delay to avoid rate limiting
            if retry_count > 0:
                delay = random.randint(5, 15) + (retry_count * 5)
                logger.info(f"Waiting {delay} seconds before retry {retry_count}...")
                time.sleep(delay)
            
            cmd = [
                'yt-dlp', 
                '--write-auto-subs', 
                '--write-subs', 
                '--sub-langs', 'en.*', 
                '--sub-format', 'vtt', 
                '--skip-download',
                '--no-warnings'  # Reduce log noise
            ]
            
            # Use random user agent to appear more like a regular browser
            user_agent = get_random_user_agent()
            cmd.extend(['--user-agent', user_agent])
            
            # Add rate limiting and retry options
            cmd.extend([
                '--sleep-interval', '1',  # Sleep between requests
                '--max-sleep-interval', '5',
                '--sleep-subtitles', '1',  # Sleep between subtitle downloads
                '--retries', '3',
                '--fragment-retries', '3'
            ])
            
            # Disable impersonation to avoid the warning/error
            cmd.extend(['--extractor-args', 'youtube:player_client=web'])
            
            if proxy_url:
                cmd.extend(['--proxy', proxy_url])
            
            if cookies_file_path:
                cmd.extend(['--cookies', cookies_file_path])

            cmd.extend(['--output', f'{temp_dir}/%(id)s.%(ext)s', youtube_url])

            logger.info(f"Running yt-dlp with enhanced rate limiting (attempt {retry_count + 1})...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,  # Increased timeout
                check=True, 
                encoding='utf-8'
            )
            
            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            if not vtt_files: 
                raise RuntimeError("No English subtitles were found for this video.")
            
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            return parse_vtt_to_structured_transcript(vtt_content)

        except subprocess.TimeoutExpired:
            raise RuntimeError("The transcript download timed out (120s).")
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp failed: {e.stderr}")
            
            # Handle specific error cases
            if "429" in e.stderr or "Too Many Requests" in e.stderr:
                if retry_count < 3:  # Allow up to 3 retries
                    logger.info(f"Rate limited (429), retrying... (attempt {retry_count + 1}/3)")
                    return get_transcript_with_ytdlp(youtube_url, retry_count + 1)
                else:
                    raise RuntimeError("YouTube is rate limiting requests. Please try again later (after 15-30 minutes).")
            
            if "cookies are no longer valid" in e.stderr:
                raise RuntimeError("Could not fetch subtitles: Your provided cookies have expired. Please refresh them.")
            
            if "Video unavailable" in e.stderr:
                raise RuntimeError("This video is unavailable or has been removed.")
            
            if "Private video" in e.stderr:
                raise RuntimeError("This video is private and cannot be accessed.")
            
            # Generic fallback
            raise RuntimeError(f"Unable to fetch subtitles. Error: {e.stderr.split('ERROR:')[-1].strip() if 'ERROR:' in e.stderr else 'Unknown error'}")

def summarize_with_google_ai(transcript: str, word_count: int) -> str:
    if not model: 
        raise RuntimeError("AI model is not available due to a configuration error.")
    
    prompt = f"""As an expert analyst, provide a comprehensive summary of the following video transcript. 
    
Format your output in Markdown with two distinct sections:

## Quick Summary
A concise paragraph (2-3 sentences) capturing the main point and conclusion.

## Key Takeaways
A bulleted list of the 4-6 most important points, insights, or conclusions from the video.

---
Transcript ({word_count} words): "{transcript}"
"""
    
    try:
        response = model.generate_content(prompt)
        summary_markdown = response.text
        return summary_markdown
    except Exception as e:
        logger.error(f"AI summarization error: {e}")
        raise RuntimeError(f"The AI summarization failed. Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "YouTube Summarizer API is running", "status": "healthy"}

@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    try:
        logger.info(f"Processing summarization request for: {request.youtube_url}")
        
        structured_transcript, plain_text = get_transcript_with_ytdlp(request.youtube_url)
        word_count = len(plain_text.split())
        
        if word_count < 50:
            return JSONResponse(content={
                "summary": "This video is too short to summarize meaningfully.", 
                "transcript": structured_transcript,
                "word_count": word_count
            })
        
        summary_markdown = summarize_with_google_ai(plain_text, word_count)
        
        return JSONResponse(content={
            "summary": summary_markdown, 
            "transcript": structured_transcript,
            "word_count": word_count
        })
        
    except (ValueError, RuntimeError) as e:
        logger.error(f"Client error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred. Please try again later."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
