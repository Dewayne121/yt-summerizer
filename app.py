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
app = FastAPI(title="YouTube Summarizer API", debug=True)

# Add debugging middleware
@app.middleware("http")
async def debug_middleware(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
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

def clean_youtube_url(url: str) -> str:
    """Clean YouTube URL by removing unnecessary parameters"""
    video_id = get_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def get_transcript_with_ytdlp(youtube_url: str, retry_count: int = 0) -> Tuple[list, str]:
    if not get_video_id(youtube_url): 
        raise ValueError("Invalid YouTube URL format provided.")

    # Clean the URL first
    clean_url = clean_youtube_url(youtube_url)
    logger.info(f"Cleaned URL: {clean_url}")

    proxy_url = os.getenv('PROXY_URL')
    cookies_data = os.getenv('YOUTUBE_COOKIES')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cookies_file_path = None
        if cookies_data:
            netscape_formatted_cookies = convert_json_to_netscape(cookies_data)
            cookies_file_path = os.path.join(temp_dir, 'cookies.txt')
            with open(cookies_file_path, 'w', encoding='utf-8') as f:
                f.write(netscape_formatted_cookies)
        
        # Try multiple subtitle strategies with different YouTube client configurations
        subtitle_strategies = [
            # Strategy 1: Use web client with auto-generated English subtitles
            {'sub_langs': 'en', 'write_auto_subs': True, 'write_subs': False, 'client': 'web'},
            # Strategy 2: Use android client (sometimes works when web doesn't)
            {'sub_langs': 'en', 'write_auto_subs': True, 'write_subs': False, 'client': 'android'},
            # Strategy 3: Use ios client
            {'sub_langs': 'en', 'write_auto_subs': True, 'write_subs': False, 'client': 'ios'},
            # Strategy 4: Minimal command - just basics
            {'sub_langs': 'en', 'write_auto_subs': True, 'write_subs': False, 'minimal': True},
            # Strategy 5: Use web client without proxy
            {'sub_langs': 'en', 'write_auto_subs': True, 'write_subs': False, 'client': 'web', 'no_proxy': True},
            # Strategy 6: Try TV embedded client
            {'sub_langs': 'en', 'write_auto_subs': True, 'write_subs': False, 'client': 'tv:embed'},
            # Strategy 7: All languages as fallback
            {'sub_langs': 'all', 'write_auto_subs': True, 'write_subs': True, 'client': 'web'},
        ]
        
        for strategy_idx, strategy in enumerate(subtitle_strategies):
            try:
                # Add random delay to avoid rate limiting
                if retry_count > 0:
                    delay = random.randint(5, 15) + (retry_count * 5)
                    logger.info(f"Waiting {delay} seconds before retry {retry_count}...")
                    time.sleep(delay)
                
                cmd = ['yt-dlp', '--sub-format', 'vtt', '--skip-download', '--no-warnings']
                
                # Add subtitle strategy options
                if strategy['write_auto_subs']:
                    cmd.append('--write-auto-subs')
                if strategy['write_subs']:
                    cmd.append('--write-subs')
                cmd.extend(['--sub-langs', strategy['sub_langs']])
                
                # Check if this is a minimal strategy
                if strategy.get('minimal', False):
                    # Just add the basic options and skip everything else
                    pass
                else:
                    # Use different YouTube client configurations
                    client = strategy.get('client', 'web')
                    cmd.extend(['--extractor-args', f'youtube:player_client={client}'])
                    
                    # Use random user agent to appear more like a regular browser
                    user_agent = get_random_user_agent()
                    cmd.extend(['--user-agent', user_agent])
                    
                    # Add rate limiting and retry options
                    cmd.extend([
                        '--sleep-interval', '2',  # Increased sleep
                        '--max-sleep-interval', '10',
                        '--sleep-subtitles', '2',
                        '--retries', '5',  # More retries
                        '--fragment-retries', '5'
                    ])
                    
                    # Conditionally add proxy (some strategies skip it)
                    use_proxy = not strategy.get('no_proxy', False)
                    if proxy_url and use_proxy:
                        cmd.extend(['--proxy', proxy_url])
                    elif not use_proxy:
                        logger.info("Skipping proxy for this strategy")
                    
                    if cookies_file_path:
                        cmd.extend(['--cookies', cookies_file_path])

                cmd.extend(['--output', f'{temp_dir}/%(id)s.%(ext)s', clean_url])

                logger.info(f"Trying subtitle strategy {strategy_idx + 1}/{len(subtitle_strategies)}: client={strategy.get('client', 'minimal' if strategy.get('minimal') else 'web')}, langs={strategy['sub_langs']}, no_proxy={strategy.get('no_proxy', False)} (attempt {retry_count + 1})...")
                logger.info(f"Command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=120,  # Increased timeout
                    check=False,  # Don't raise exception on non-zero exit
                    encoding='utf-8'
                )
                
                # Log the result for debugging
                logger.info(f"yt-dlp exit code: {result.returncode}")
                if result.stdout:
                    logger.info(f"yt-dlp stdout: {result.stdout[:500]}...")
                if result.stderr:
                    logger.info(f"yt-dlp stderr: {result.stderr[:500]}...")
                
                # Look for any files created (not just VTT)
                all_files = glob.glob(f"{temp_dir}/*")
                logger.info(f"Files created in temp dir: {[os.path.basename(f) for f in all_files]}")
                
                # Look for any VTT files
                vtt_files = glob.glob(f"{temp_dir}/*.vtt")
                if vtt_files:
                    # Prioritize English files
                    english_files = [f for f in vtt_files if any(lang in f.lower() for lang in ['en', 'english'])]
                    chosen_file = english_files[0] if english_files else vtt_files[0]
                    
                    logger.info(f"Found subtitle file: {os.path.basename(chosen_file)}")
                    with open(chosen_file, 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    
                    # Validate that we have actual content
                    structured_transcript, plain_text = parse_vtt_to_structured_transcript(vtt_content)
                    if len(plain_text.strip()) > 10:  # Must have at least some content
                        return structured_transcript, plain_text
                    else:
                        logger.warning(f"Subtitle file {chosen_file} had insufficient content, trying next strategy...")
                        continue
                
                # If no files found, try next strategy
                logger.warning(f"No subtitle files found with strategy {strategy_idx + 1}, trying next...")
                
                # If this failed and it's not the last strategy, continue
                if result.returncode != 0 and strategy_idx < len(subtitle_strategies) - 1:
                    continue
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Strategy {strategy_idx + 1} timed out")
                if strategy_idx < len(subtitle_strategies) - 1:
                    continue
                else:
                    raise RuntimeError("The transcript download timed out (120s).")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Strategy {strategy_idx + 1} failed with CalledProcessError: {e.stderr}")
                if strategy_idx < len(subtitle_strategies) - 1:
                    continue  # Try next strategy
                else:
                    # This was the last strategy, handle the error
                    if "429" in e.stderr or "Too Many Requests" in e.stderr:
                        if retry_count < 3:
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
            except Exception as e:
                logger.error(f"Strategy {strategy_idx + 1} failed with unexpected error: {e}")
                if strategy_idx < len(subtitle_strategies) - 1:
                    continue
                else:
                    raise RuntimeError(f"Failed to extract subtitles: {str(e)}")
        
        # If we get here, none of the strategies worked
        raise RuntimeError("No subtitles found for this video. The video may not have captions available, or they may be in a language other than English.")

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

def check_available_subtitles(youtube_url: str) -> dict:
    """Debug function to check what subtitles are available for a video"""
    video_id = get_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}
    
    try:
        # First, try to list available subtitles
        cmd = [
            'yt-dlp', 
            '--list-subs',
            '--no-warnings',
            youtube_url
        ]
        
        list_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding='utf-8')
        
        # Then try to get basic video info
        info_cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-warnings',
            youtube_url
        ]
        
        info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30, encoding='utf-8')
        
        video_info = {}
        if info_result.returncode == 0:
            try:
                video_data = json.loads(info_result.stdout)
                video_info = {
                    "title": video_data.get("title", "Unknown"),
                    "duration": video_data.get("duration", "Unknown"),
                    "uploader": video_data.get("uploader", "Unknown"),
                    "has_subtitles": bool(video_data.get("subtitles", {})),
                    "has_automatic_captions": bool(video_data.get("automatic_captions", {})),
                    "available_subtitle_languages": list(video_data.get("subtitles", {}).keys()) if video_data.get("subtitles") else [],
                    "available_auto_caption_languages": list(video_data.get("automatic_captions", {}).keys()) if video_data.get("automatic_captions") else []
                }
            except json.JSONDecodeError:
                video_info = {"error": "Could not parse video info"}
        
        return {
            "video_id": video_id,
            "video_info": video_info,
            "list_subs_output": list_result.stdout,
            "list_subs_stderr": list_result.stderr,
            "list_subs_returncode": list_result.returncode,
            "info_stderr": info_result.stderr if info_result.returncode != 0 else None
        }
    except Exception as e:
        return {"error": f"Failed to check subtitles: {str(e)}"}

# --- Route Definitions ---
@app.get("/")
async def root():
    return {"message": "YouTube Summarizer API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path, 
                "methods": list(route.methods)
            })
    return {"routes": routes}

@app.get("/api/debug-subtitles/")
async def debug_subtitles(url: str):
    """Debug endpoint to check available subtitles for a video"""
    try:
        result = check_available_subtitles(url)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Define the main summarize endpoint
async def summarize_video(request: SummarizeRequest):
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

# Register the endpoint with multiple routes
app.post("/api/summarize/")(summarize_video)
app.post("/api/summarize")(summarize_video)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
