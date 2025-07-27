import gradio as gr
from transformers import pipeline
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSM.py` file on GitHub.

```python
import gradio as gr
from transformers import pipeline
import os
fromiddleware
from pydantic import BaseModel
import re
import logging
from contextlib import asynccontextmanager

 fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)from pydantic import BaseModel
import re
import logging
from contextlib import asynccontextmanager

# --- Setup

# --- Global variables for the model ---
summarizer = None
model_load_error = None

# ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- --- Lifespan Manager (Modern FastAPI Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app Global variables for the model ---
summarizer = None
model_load_error = None

# --- Lifespan Manager (: FastAPI):
    """Handles application startup events: loading the model."""
    global summarizer, model_load_errorModern FastAPI Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    logger.info("Lifespan: Application startup...")
    
    # Load the machine learning model
Handles application startup events: loading the model."""
    global summarizer, model_load_error
    logger.info("Lifespan: Application startup...")
    
    # Load the machine learning model
    try:
    try:
        model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
        logger.info(f"Lifespan: Loading model '{        model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-model_name}'...")
        summarizer = pipeline(
            "summarization",
            model=model_12-6")
        logger.info(f"Lifespan: Loading model '{model_name}'...")
        summarizer = pipeline(
            "summarization",
            model=model_name,
            name,
            device=-1  # Force CPU usage
        )
        logger.info(f"Lifespan: Model loaded successfully.")
    except Exception as e:
        model_load_error = str(device=-1  # Force CPU usage
        )
        logger.info(f"Lifespan: Modele)
        logger.error(f"Lifespan: FATAL - Failed to load model: {e loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        }", exc_info=True)
    
    yield  # The application is now running
    
    #logger.error(f"Lifespan: FATAL - Failed to load model: {e}", exc_info=True)
    
    yield  # The application is now running
    
    # --- Shutdown logic ( --- Shutdown logic (if any) would go here ---
    logger.info("Lifespan: Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(title="YouTube Video Summarizer API", lifespan=lifesif any) would go here ---
    logger.info("Lifespan: Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(title="YouTube Video Summarizer API", lifespan=lifespan)

#pan)

# Add CORS middleware to allow requests from any origin (e.g., your front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow Add CORS middleware to allow requests from any origin (e.g., your front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dynamic Import# --- Dynamic Import of YouTube API ---
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound of YouTube API ---
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
except ImportError:
except ImportError:
    YouTubeTranscriptApi = None
    NoTranscriptFound = type('NoTranscriptFound', (Exception,), {}) # Dummy class

# --- Pydantic Model for API validation ---
class SummarizeRequest(Base
    YouTubeTranscriptApi = None
    NoTranscriptFound = type('NoTranscriptFound', (Exception,), {}) #Model):
    youtube_url: str

# --- Core Logic Functions ---
def get_video_id( Dummy class

# --- Pydantic Model for API validation ---
class SummarizeRequest(BaseModel):
    url: str):
    """Extracts the YouTube video ID from various URL formats."""
    if not url:youtube_url: str

# --- Core Logic Functions ---
def get_video_id(url: str):
    """Extracts the YouTube video ID from various URL formats."""
    if not url: return None
     return None
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.compatterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9v=([a-zA-Z0-9_-]{11})',
        r'(?:https?_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11pattern, url.strip())
        if match: return match.group(1)
    return None

def})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match: return match.group(1)
    return None

def process_and_ process_and_summarize(youtube_url: str):
    """The main business logic for summarizing a YouTubesummarize(youtube_url: str):
    """The main business logic for summarizing a YouTube video."""
     video."""
    if summarizer is None:
        raise RuntimeError(f"Model is not available. Load error: {model_load_error or 'Unknown reason.'}")
    if YouTubeTranscriptApi is None:
        raiseif summarizer is None:
        raise RuntimeError(f"Model is not available. Load error: {model_load_error or 'Unknown reason.'}")
    if YouTubeTranscriptApi is None:
        raise ModuleNotFoundError("The ModuleNotFoundError("The 'youtube-transcript-api' library is not installed on the server.")

    video_id 'youtube-transcript-api' library is not installed on the server.")

    video_id = get_video = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL formatInvalid YouTube URL format provided.")

    try:
        # ================================================================= #
        # === THE provided.")

    try:
        # --- THIS IS THE MODIFIED LINE ---
        # Added cookies=None to handle FIX: Added `cookies=None` to bypass YouTube's consent page === #
        # ================================================================= YouTube's consent screen on servers.
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages #
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'], cookies=['en'], cookies=None)
        full_transcript = " ".join([item['text'] for item in=None)
        
        full_transcript = " ".join([item['text'] for item in transcript_ transcript_list])
    except NoTranscriptFound:
        raise NoTranscriptFound("An English transcript was not found for this videolist])
    except NoTranscriptFound:
        raise NoTranscriptFound("An English transcript was not found for this.")
    except Exception as e:
        logger.error(f"Error fetching transcript for video_id '{ video.")
    except Exception as e:
        logger.error(f"Error fetching transcript for video_idvideo_id}': {e}")
        raise RuntimeError("Failed to retrieve the transcript from YouTube.")

    if len(full '{video_id}': {e}")
        raise RuntimeError("Failed to retrieve the transcript from YouTube.")

    if len(_transcript.split()) < 50:
        raise ValueError("Transcript is too short to generate a meaningful summaryfull_transcript.split()) < 50:
        raise ValueError("Transcript is too short to generate a meaningful.")

    # Chunking and summarizing
    words = full_transcript.split()
    chunks = [" ". summary.")

    # Chunking and summarizing
    words = full_transcript.split()
    chunks = [" ".join(words[i:i + 512]) for i in range(0, len(wordsjoin(words[i:i + 512]) for i in range(0, len(words), 512)]
    
    max_chunks_for_demo = 3 
    summaries), 512)]
    
    max_chunks_for_demo = 3 
    summar = [summarizer(chunk, max_length=130, min_length=30, do_ies = [summarizer(chunk, max_length=130, min_length=30, dosample=False)[0]['summary_text'] for chunk in chunks[:max_chunks_for_demo]]
_sample=False)[0]['summary_text'] for chunk in chunks[:max_chunks_for_demo]]    
    if not summaries:
        raise RuntimeError("Model failed to generate a summary from the transcript.")

    
    
    if not summaries:
        raise RuntimeError("Model failed to generate a summary from the transcript.")

    final_final_summary = "\n\n".join([f"• {s}" for s in summaries])
    summary = "\n\n".join([f"• {s}" for s in summaries])
    note =note = f"\n\n*(Note: Summary generated from the first {len(summaries)} part(s f"\n\n*(Note: Summary generated from the first {len(summaries)} part(s) of) of the video.)*" if len(chunks) > len(summaries) else ""
    return final_ the video.)*" if len(chunks) > len(summaries) else ""
    return final_summary

summary

# --- API Endpoint for Programmatic Access ---
@app.post("/api/summarize/")
async# --- API Endpoint for Programmatic Access ---
@app.post("/api/summarize/")
async def api def api_summarize(request: SummarizeRequest):
    """This endpoint is for your custom front-end_summarize(request: SummarizeRequest):
    """This endpoint is for your custom front-end or other or other services to call."""
    try:
        summary = process_and_summarize(request.youtube services to call."""
    try:
        summary = process_and_summarize(request.youtube_url_url)
        return JSONResponse(content={"summary": summary})
    except (ValueError, NoTranscriptFound)
        return JSONResponse(content={"summary": summary})
    except (ValueError, NoTranscriptFound, Module, ModuleNotFoundError) as e:
        return JSONResponse(status_code=400, content={"errorNotFoundError) as e:
        return JSONResponse(status_code=400, content={"error": str": str(e)}) # User/Client Error
    except Exception as e:
        logger.error(f(e)}) # User/Client Error
    except Exception as e:
        logger.error(f"API"API Error: {e}", exc_info=True)
        return JSONResponse(status_code=5 Error: {e}", exc_info=True)
        return JSONResponse(status_code=50000, content={"error": "An internal server error occurred."})

# --- Gradio Interface ---
with, content={"error": "An internal server error occurred."})

# --- Gradio Interface ---
with gr. gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Enter a YouTube URL togr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Enter a YouTube URL to get an get an AI-generated summary. The video must have English captions/transcripts.")
    
    with gr. AI-generated summary. The video must have English captions/transcripts.")
    
    with gr.Row():Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.youtube.com/watch?v=...", lines=1, scale=3)
        submit_btn = grcom/watch?v=...", lines=1, scale=3)
        submit_btn = gr.Button.Button("Summarize", variant="primary", scale=1)
    
    output = gr.Markdown(("Summarize", variant="primary", scale=1)
    
    output = gr.Markdown(label="label="Summary")

    def gradio_summarize_wrapper(youtube_url: str):
        """Wrapper function for theSummary")

    def gradio_summarize_wrapper(youtube_url: str):
        """Wrapper function for Gradio UI to provide clean error messages."""
        try:
            # Check for model availability before processing
             the Gradio UI to provide clean error messages."""
        try:
            # Check for model availability before processing
if summarizer is None:
                 return f"❌ **Error:** Model is not ready. Please try again in            if summarizer is None:
                 return f"❌ **Error:** Model is not ready. Please try again a moment. {model_load_error or ''}"
            
            summary = process_and_summarize in a moment. {model_load_error or ''}"
            
            summary = process_and_summar(youtube_url)
            return f"✅ **Summary:**\n\n{summary}"
        exceptize(youtube_url)
            return f"✅ **Summary:**\n\n{summary}"
         Exception as e:
            # Return any error from the core logic directly to the user
            return f"❌except Exception as e:
            # Return any error from the core logic directly to the user
            return f" **Error:** {e}"

    submit_btn.click(fn=gradio_summarize_wrapper,❌ **Error:** {e}"

    submit_btn.click(fn=gradio_summarize_wrapper inputs=[url_input], outputs=[output])
    
    gr.Examples(
        examples=[["https, inputs=[url_input], outputs=[output])
    
    gr.Examples(
        examples=[["://www.youtube.com/watch?v=jNQXAC9IVRw"], ["https://www.https://www.youtube.com/watch?v=jNQXAC9IVRw"], ["https://www.youtube.com/watch?v=9bZkp7q19f0"]],
        inputsyoutube.com/watch?v=9bZkp7q19f0"]],
        inputs=[url_input=[url_input],
        outputs=[output],
        fn=gradio_summarize_wrapper
    )],
        outputs=[output],
        fn=gradio_summarize_wrapper
    )

# Mount the Gradio UI onto the FastAPI app
app = gr.mount_gradio_app(app, gradio_interface

# Mount the Gradio UI onto the FastAPI app
app = gr.mount_gradio_app(app, path="/")
```, gradio_interface, path="/")
