import gradio as gr
from transformers import pipeline
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import logging
from contextlib import asynccontextmanager

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global variables for the model ---
summarizer = None
model_load_error = None

# --- Lifespan Manager (Modern FastAPI Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup events: loading the model."""
    global summarizer, model_load_error
    logger.info("Lifespan: Application startup...")
    
    # Load the machine learning model
    try:
        model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
        logger.info(f"Lifespan: Loading model '{model_name}'...")
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1  # Force CPU usage
        )
        logger.info(f"Lifespan: Model loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Lifespan: FATAL - Failed to load model: {e}", exc_info=True)
    
    yield  # The application is now running
    
    # --- Shutdown logic (if any) would go here ---
    logger.info("Lifespan: Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(title="YouTube Video Summarizer API", lifespan=lifespan)

# Add CORS middleware to allow requests from any origin (e.g., your front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dynamic Import of YouTube API ---
# This makes error handling cleaner if the library is missing.
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
except ImportError:
    YouTubeTranscriptApi = None
    NoTranscriptFound = type('NoTranscriptFound', (Exception,), {}) # Dummy class

# --- Pydantic Model for API validation ---
class SummarizeRequest(BaseModel):
    youtube_url: str

# --- Core Logic Functions ---
def get_video_id(url: str):
    """Extracts the YouTube video ID from various URL formats."""
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
    """The main business logic for summarizing a YouTube video."""
    if summarizer is None:
        raise RuntimeError(f"Model is not available. Load error: {model_load_error or 'Unknown reason.'}")
    if YouTubeTranscriptApi is None:
        raise ModuleNotFoundError("The 'youtube-transcript-api' library is not installed on the server.")

    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format provided.")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'], cookies=None)
        full_transcript = " ".join([item['text'] for item in transcript_list])
    except NoTranscriptFound:
        raise NoTranscriptFound("An English transcript was not found for this video.")
    except Exception as e:
        logger.error(f"Error fetching transcript for video_id '{video_id}': {e}")
        raise RuntimeError("Failed to retrieve the transcript from YouTube.")

    if len(full_transcript.split()) < 50:
        raise ValueError("Transcript is too short to generate a meaningful summary.")

    # Chunking and summarizing
    words = full_transcript.split()
    chunks = [" ".join(words[i:i + 512]) for i in range(0, len(words), 512)]
    
    max_chunks_for_demo = 3 
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks[:max_chunks_for_demo]]
    
    if not summaries:
        raise RuntimeError("Model failed to generate a summary from the transcript.")

    final_summary = "\n\n".join([f"• {s}" for s in summaries])
    note = f"\n\n*(Note: Summary generated from the first {len(summaries)} part(s) of the video.)*" if len(chunks) > len(summaries) else ""
    return final_summary

# --- API Endpoint for Programmatic Access ---
@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    """This endpoint is for your custom front-end or other services to call."""
    try:
        summary = process_and_summarize(request.youtube_url)
        return JSONResponse(content={"summary": summary})
    except (ValueError, NoTranscriptFound, ModuleNotFoundError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)}) # User/Client Error
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})

# --- Gradio Interface ---
with gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Enter a YouTube URL to get an AI-generated summary. The video must have English captions/transcripts.")
    
    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", lines=1, scale=3)
        submit_btn = gr.Button("Summarize", variant="primary", scale=1)
    
    output = gr.Markdown(label="Summary")

    def gradio_summarize_wrapper(youtube_url: str):
        """Wrapper function for the Gradio UI to provide clean error messages."""
        try:
            # Check for model availability before processing
            if summarizer is None:
                 return f"❌ **Error:** Model is not ready. Please try again in a moment. {model_load_error or ''}"
            
            summary = process_and_summarize(youtube_url)
            return f"✅ **Summary:**\n\n{summary}"
        except Exception as e:
            # Return any error from the core logic directly to the user
            return f"❌ **Error:** {e}"

    submit_btn.click(fn=gradio_summarize_wrapper, inputs=[url_input], outputs=[output])
    
    gr.Examples(
        examples=[["https://www.youtube.com/watch?v=jNQXAC9IVRw"], ["https://www.youtube.com/watch?v=9bZkp7q19f0"]],
        inputs=[url_input]
    )

# Mount the Gradio UI onto the FastAPI app
app = gr.mount_gradio_app(app, gradio_interface, path="/")
