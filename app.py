import gradio as gr
from transformers import pipeline
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import logging
import asyncio
import threading
import requests

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global state variables ---
summarizer = None
model_load_error = None
is_model_loading = False

# --- Model Loading Logic ---
def load_model_proc():
    """The actual procedure to load the model. This is run in a background thread."""
    global summarizer, model_load_error, is_model_loading
    
    if summarizer or is_model_loading:
        return

    is_model_loading = True
    logger.info("Background thread: Starting model loading...")
    try:
        model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1
        )
        logger.info(f"Background thread: Model '{model_name}' loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Background thread: FATAL - Failed to load model: {e}", exc_info=True)
    finally:
        is_model_loading = False


# --- FastAPI App Initialization ---
app = FastAPI(title="YouTube Video Summarizer API")

@app.on_event("startup")
def startup_event():
    """Starts the model download in a separate thread so it doesn't block the server."""
    logger.info("Application startup: Triggering background model load.")
    thread = threading.Thread(target=load_model_proc)
    thread.start()


# --- Standard App Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
except ImportError:
    YouTubeTranscriptApi = None
    NoTranscriptFound = type('NoTranscriptFound', (Exception,), {})

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

def process_and_summarize(youtube_url: str):
    if is_model_loading:
        raise RuntimeError("The AI model is still loading. Please try again in a few minutes.")
    if summarizer is None:
        raise RuntimeError(f"Model is not available. Load error: {model_load_error or 'Unknown reason.'}")
    if YouTubeTranscriptApi is None:
        raise ModuleNotFoundError("The 'youtube-transcript-api' library is not installed on the server.")

    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format provided.")

    # Multiple retry strategies for fetching transcripts
    transcript_list = None
    last_error = None
    
    # Strategy 1: Try different language combinations
    language_attempts = [
        ['en'],
        ['en-US'], 
        ['en-GB'],
        ['a.en'],  # Auto-generated English
        ['en', 'en-US', 'en-GB']
    ]
    
    for languages in language_attempts:
        try:
            logger.info(f"Trying to fetch transcript with languages: {languages}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            logger.info(f"Successfully fetched transcript with languages: {languages}")
            break
        except NoTranscriptFound as e:
            last_error = e
            logger.warning(f"No transcript found for languages {languages}: {e}")
            continue
        except Exception as e:
            last_error = e
            logger.warning(f"Error with languages {languages}: {e}")
            continue
    
    # Strategy 2: If direct methods fail, try getting available transcripts first
    if transcript_list is None:
        try:
            logger.info("Trying to list available transcripts first...")
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find any English transcript
            for transcript in transcript_list_obj:
                if transcript.language_code.startswith('en') or transcript.language_code == 'a.en':
                    logger.info(f"Found transcript in language: {transcript.language_code}")
                    transcript_list = transcript.fetch()
                    break
                    
            # If no English, try the first available transcript and translate
            if transcript_list is None:
                for transcript in transcript_list_obj:
                    try:
                        logger.info(f"Trying to translate transcript from {transcript.language_code} to English")
                        transcript_list = transcript.translate('en').fetch()
                        break
                    except Exception as e:
                        logger.warning(f"Translation failed for {transcript.language_code}: {e}")
                        continue
                        
        except Exception as e:
            last_error = e
            logger.error(f"Error listing transcripts: {e}")
    
    # Strategy 3: Try with cookies (if available)
    if transcript_list is None:
        try:
            logger.info("Trying with cookies parameter...")
            # You can add cookies here if you have them
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en'],
                cookies={}  # Add cookies here if available
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Cookies approach failed: {e}")
    
    if transcript_list is None:
        if isinstance(last_error, NoTranscriptFound):
            raise NoTranscriptFound("This video has no available English transcript or subtitles.")
        else:
            logger.error(f"All transcript fetch strategies failed for video_id '{video_id}'. Last error: {last_error}", exc_info=True)
            raise RuntimeError(f"Unable to fetch transcript. This may be due to: 1) The video is private/restricted, 2) No captions are available, 3) YouTube is blocking the request. Try again later or use a different video. Error: {str(last_error)}")

    try:
        full_transcript = " ".join([item['text'] for item in transcript_list])
    except Exception as e:
        logger.error(f"Error processing transcript data: {e}", exc_info=True)
        raise RuntimeError("Error processing the transcript data.")

    if len(full_transcript.split()) < 50:
        raise ValueError("Transcript is too short to generate a meaningful summary.")

    words = full_transcript.split()
    chunks = [" ".join(words[i:i + 512]) for i in range(0, len(words), 512)]
    
    max_chunks_for_demo = 3 
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks[:max_chunks_for_demo]]
    
    if not summaries:
        raise RuntimeError("Model failed to generate a summary from the transcript.")

    final_summary = "\n\n".join([f"• {s}" for s in summaries])
    note = f"\n\n*(Note: Summary generated from the first {len(summaries)} part(s) of the video.)*" if len(chunks) > len(summaries) else ""
    return final_summary + note

# --- API Endpoint ---
@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    try:
        summary = process_and_summarize(request.youtube_url)
        return JSONResponse(content={"summary": summary})
    except (ValueError, NoTranscriptFound, ModuleNotFoundError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Gradio Interface ---
with gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Enter a YouTube URL to get an AI-generated summary.")
    
    status_display = gr.Markdown("Model status: Unknown")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", lines=1, scale=3)
        submit_btn = gr.Button("Summarize", variant="primary", scale=1)
    
    output = gr.Markdown(label="Summary")

    def get_status():
        if summarizer:
            return "✅ Model is loaded and ready."
        if is_model_loading:
            return "⏳ Model is loading in the background... (this may take several minutes)"
        if model_load_error:
            return f"❌ Model failed to load: {model_load_error}"
        return "Waiting to start..."

    def gradio_summarize_wrapper(youtube_url: str):
        try:
            summary = process_and_summarize(youtube_url)
            return f"✅ **Summary:**\n\n{summary}"
        except Exception as e:
            return f"❌ **Error:** {e}"

    submit_btn.click(fn=gradio_summarize_wrapper, inputs=[url_input], outputs=[output])
    
    gradio_interface.load(get_status, None, status_display, every=2)
    
    gr.Examples(
        examples=[["https://www.youtube.com/watch?v=jNQXAC9IVRw"], ["https://www.youtube.com/watch?v=9bZkp7q19f0"]],
        inputs=[url_input], outputs=[output], fn=gradio_summarize_wrapper
    )

app = gr.mount_gradio_app(app, gradio_interface, path="/")
