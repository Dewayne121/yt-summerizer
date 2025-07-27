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

def get_video_id(url: str):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match: return match.group(1)
    return None

def parse_vtt_to_structured_transcript(vtt_content: str):
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

def get_transcript_with_ytdlp(youtube_url: str):
    video_id = get_video_id(youtube_url)
    if not video_id: raise ValueError("Invalid YouTube URL format provided.")

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
            cmd = ['yt-dlp', '--write-auto-subs', '--write-subs', '--sub-langs', 'en.*', '--sub-format', 'vtt', '--skip-download']
            
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            cmd.extend(['--user-agent', user_agent])
            
            if proxy_url:
                logger.info("Using a proxy for the request.")
                cmd.extend(['--proxy', proxy_url])
            
            if cookies_file_path:
                logger.info("Using browser cookies for the request.")
                cmd.extend(['--cookies', cookies_file_path])

            cmd.extend(['--output', f'{temp_dir}/%(id)s.%(ext)s', youtube_url])

            logger.info("Running yt-dlp with stable configuration (User-Agent, Proxy, Cookies)...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=True, encoding='utf-8')
            
            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            if not vtt_files: raise RuntimeError("No English subtitles were found for this video.")
            
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            return parse_vtt_to_structured_transcript(vtt_content)

        except subprocess.TimeoutExpired:
            raise RuntimeError("The transcript download timed out (90s). The video may be exceptionally long.")
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp failed: {e.stderr}")
            if "Sign in to confirm you’re not a bot" in e.stderr or "cookies are no longer valid" in e.stderr:
                raise RuntimeError("Could not fetch subtitles: YouTube requires a valid login. Your provided cookies may have expired. Please refresh them.")
            raise RuntimeError(f"Could not fetch subtitles. Error: {e.stderr}")

def summarize_with_google_ai(transcript: str, word_count: int):
    if not model: raise RuntimeError("AI model is not available due to a configuration error.")
    prompt = f"""As an expert analyst, provide a comprehensive summary of the following video transcript. Format your output in Markdown with two distinct sections: ## Quick Summary\nA concise paragraph capturing the main point and conclusion. ## Key Takeaways\nA bulleted list of the 4-6 most important points. --- Transcript: "{transcript}" """
    try:
        response = model.generate_content(prompt)
        summary_markdown = response.text
        return summary_markdown
    except Exception as e:
        raise RuntimeError(f"The AI summarization failed. Error: {e}")

@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    try:
        structured_transcript, plain_text = get_transcript_with_ytdlp(request.youtube_url)
        word_count = len(plain_text.split())
        if word_count < 50:
             return JSONResponse(content={"summary": "This video is too short to summarize.", "transcript": structured_transcript})
        summary_markdown = summarize_with_google_ai(plain_text, word_count)
        return JSONResponse(content={"summary": summary_markdown, "transcript": structured_transcript})
    except (ValueError, RuntimeError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"An unexpected internal error occurred: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})
