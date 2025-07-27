import gradio as gr
from transformers import pipeline
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import logging
import threading
import subprocess
import tempfile
import json
import glob
from pathlib import Path
import signal
import time
import torch  # --- NEW ---
import io      # --- NEW ---

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global state variables ---
summarizer = None
model_load_error = None
is_model_loading = False
device = "cpu" # --- NEW --- Default to CPU

# --- Model Loading Logic ---
def load_model_proc():
    """The actual procedure to load the model. This is run in a background thread."""
    global summarizer, model_load_error, is_model_loading, device

    if summarizer or is_model_loading:
        return

    is_model_loading = True
    logger.info("Background thread: Starting model loading...")
    try:
        # --- NEW: GPU ACCELERATION ---
        # Automatically use GPU if available for a 10-20x speedup in summarization.
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA (GPU) is available! Using GPU for inference.")
        else:
            device = "cpu"
            logger.info("CUDA (GPU) not available. Using CPU for inference.")

        model_options = [
            "sshleifer/distilbart-cnn-6-6",
            "sshleifer/distilbart-cnn-12-6",
            "facebook/bart-large-cnn",
        ]
        model_name = os.getenv("MODEL_NAME", model_options[0])
        if model_name not in model_options:
            model_options.insert(0, model_name)

        for model in model_options:
            try:
                logger.info(f"Trying to load model: {model} onto device: {device}")
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    device=device, # --- MODIFIED ---
                    # --- NEW: Batching for faster inference ---
                    batch_size=4 if device == "cuda" else 2
                )
                logger.info(f"Successfully loaded model: {model}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
                if model == model_options[-1]:
                    raise e
                continue

    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Background thread: FATAL - Failed to load any model: {e}", exc_info=True)
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
        if (line.startswith('WEBVTT') or line.startswith('Kind:') or
            line.startswith('Language:') or '-->' in line or
            line.isdigit() or not line):
            continue
        line = re.sub(r'<[^>]+>', '', line)
        if line:
            text_lines.append(line)
    return ' '.join(text_lines)

# --- NEW: MODIFIED FOR REAL-TIME STREAMING ---
def get_transcript_with_ytdlp(youtube_url: str):
    """
    Get transcript using yt-dlp, yielding progress updates. This is a generator.
    """
    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format provided.")

    with tempfile.TemporaryDirectory() as temp_dir:
        process = None
        try:
            cmd = [
                'yt-dlp', '--write-auto-subs', '--write-subs', '--sub-langs', 'en.*',
                '--sub-format', 'vtt', '--skip-download',
                '--output', f'{temp_dir}/%(id)s.%(ext)s', youtube_url
            ]
            logger.info(f"Running yt-dlp command: {' '.join(cmd)}")
            yield "⏳ Fetching video metadata..."

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

            # Stream stdout for progress updates
            for line in iter(process.stdout.readline, ''):
                if "[youtube]" in line and "Downloading webpage" in line:
                    yield "➡️ Downloading video information..."
                elif "has received" in line:
                    yield "➡️ Receiving subtitle data..."
                logger.info(f"yt-dlp stdout: {line.strip()}")

            process.wait(timeout=60) # Wait for the process to finish

            if process.returncode != 0:
                stderr_output = process.stderr.read()
                logger.error(f"yt-dlp failed with return code {process.returncode}")
                logger.error(f"Error output: {stderr_output}")
                raise RuntimeError(f"Failed to fetch subtitles: {stderr_output}")

            yield "✅ Transcript downloaded. Parsing..."

            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            if not vtt_files:
                raise RuntimeError("No subtitle files were downloaded. Captions may not be available.")

            subtitle_file = vtt_files[0]
            logger.info(f"Using subtitle file: {subtitle_file}")

            with open(subtitle_file, 'r', encoding='utf-8') as f:
                vtt_content = f.read()

            transcript_text = parse_vtt_content(vtt_content)
            if not transcript_text or len(transcript_text.split()) < 20:
                raise RuntimeError("Extracted transcript is too short or empty.")

            yield f"✅ Transcript parsed ({len(transcript_text.split()):,} words)."
            yield transcript_text # The final yield is the transcript itself

        except subprocess.TimeoutExpired:
            raise RuntimeError("Transcript fetch timed out. Please try again.")
        finally:
            if process:
                process.kill()


def create_extractive_summary(text, num_sentences=8):
    """Creates a fast extractive summary."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Simple scoring: prioritize first and last sentences
    scored = [(s, i) for i, s in enumerate(sentences)]
    first_part = scored[:int(num_sentences * 0.7)]
    last_part = scored[-int(num_sentences * 0.3):]
    
    selection = sorted(first_part + last_part, key=lambda x: x[1])
    summary_sentences = [s for s, i in selection]
    return ' '.join(summary_sentences)


def format_summary(title, summary_points, total_words):
    """Unified formatting for summaries."""
    formatted_summary = f"## {title}\n\n"
    for point in summary_points:
        formatted_summary += f"• {point}\n\n"
    formatted_summary += f"*📊 Processed approximately {total_words:,} words.*"
    return formatted_summary

# --- NEW: REWRITTEN FOR SPEED AND STREAMING ---
def process_and_summarize(youtube_url: str):
    """
    Main processing generator. Yields status updates and final summaries.
    """
    # 1. Get Transcript (with streaming updates)
    transcript_generator = get_transcript_with_ytdlp(youtube_url)
    full_transcript = ""
    for update in transcript_generator:
        if update.startswith("✅") or update.startswith("⏳") or update.startswith("➡️"):
            yield update # Pass status updates to the UI
        else:
            full_transcript = update # This is the final transcript text

    if not full_transcript:
        raise RuntimeError("Failed to retrieve a valid transcript.")
    
    words = full_transcript.split()
    word_count = len(words)
    
    if word_count < 100:
        yield "⚠️ Transcript is too short for a detailed summary. Displaying full text."
        yield f"## 📜 Full Transcript (Short Video)\n\n{full_transcript}"
        return

    # 2. Generate INSTANT Quick Summary (Extractive)
    yield "🧠 Generating Quick Summary..."
    quick_summary_text = create_extractive_summary(full_transcript, num_sentences=8)
    quick_summary_formatted = format_summary("⚡ Quick Summary (Key Sentences)", quick_summary_text.split('. '), word_count)
    yield quick_summary_formatted # Display this first

    # 3. Check if AI model is available
    if is_model_loading:
        yield quick_summary_formatted + "\n\n---\n\n*⏳ AI model is still loading. AI summary will be available soon.*"
        return
    if not summarizer:
        yield quick_summary_formatted + f"\n\n---\n\n*❌ AI model failed to load ({model_load_error}). Only Quick Summary is available.*"
        return

    # 4. Generate AI Summary (using batching for speed)
    yield quick_summary_formatted + "\n\n---\n\n" + "🤖 **Generating detailed AI summary (this may take a moment)...**"
    
    # Smart chunking
    chunk_size = 400
    overlap = 50
    chunks = [
        " ".join(words[i:i + chunk_size]) 
        for i in range(0, word_count, chunk_size - overlap)
    ]
    
    # Cap the number of chunks to summarize to keep it fast
    max_chunks = 4 if device == "cuda" else 2 # More chunks if on GPU
    chunks_to_process = chunks[:max_chunks]
    
    try:
        # BATCH aI INFERENCE - much faster than a loop
        logger.info(f"Summarizing {len(chunks_to_process)} chunks in a single batch...")
        ai_summaries = summarizer(
            chunks_to_process,
            max_length=130,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        summary_points = [s['summary_text'] for s in ai_summaries]
        
        # Combine quick summary with AI summary
        ai_summary_formatted = format_summary("🤖 Detailed AI Summary", summary_points, word_count)
        final_output = quick_summary_formatted + "\n\n---\n\n" + ai_summary_formatted
        yield final_output

    except Exception as e:
        logger.error(f"AI summarization failed: {e}", exc_info=True)
        yield quick_summary_formatted + "\n\n---\n\n" + f"❌ An error occurred during AI summarization: {e}"


# --- API Endpoint (MODIFIED to handle streaming for potential future use) ---
@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    try:
        # For a simple JSON API, we collect all results from the generator
        final_result = ""
        for result in process_and_summarize(request.youtube_url):
            final_result = result # Keep the last update
        return JSONResponse(content={"summary": final_result})
    except (ValueError, RuntimeError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})

# --- Gradio Interface (MODIFIED for streaming output) ---
with gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Get a **quick, extractive summary almost instantly**, followed by a **detailed AI-generated summary**.")

    status_display = gr.Markdown("Model status: Unknown")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", lines=1, scale=3)
        submit_btn = gr.Button("🚀 Summarize", variant="primary", scale=1)

    output = gr.Markdown(label="Summary", value="Your summary will appear here...")

    def get_status():
        if summarizer:
            return f"✅ **Model Status:** Ready ({device.upper()})"
        if is_model_loading:
            return "⏳ **Model Status:** Loading in background..."
        if model_load_error:
            return f"❌ **Model Status:** Failed to load - {model_load_error}"
        return "⏳ **Model Status:** Initializing..."

    # --- MODIFIED: The wrapper is now a generator to stream updates to the UI ---
    def gradio_summarize_wrapper(youtube_url: str):
        if not youtube_url:
            yield "❌ **Error:** Please enter a YouTube URL."
            return
            
        try:
            # The `process_and_summarize` function is a generator, so we loop through its yields
            # and yield them again to the Gradio interface.
            yield from process_and_summarize(youtube_url)
        except Exception as e:
            logger.error(f"Gradio Error: {e}", exc_info=True)
            yield f"❌ **An unexpected error occurred:** {str(e)}"

    # When the output is a generator, Gradio automatically updates the output component for each yield
    submit_btn.click(fn=gradio_summarize_wrapper, inputs=[url_input], outputs=[output])

    gradio_interface.load(get_status, None, status_display, every=3)

    gr.Examples(
        examples=[
            ["https://www.youtube.com/watch?v=jNQXAC9IVRw"], # MrBeast
            ["https://www.youtube.com/watch?v=9bZkp7q19f0"], # MKBHD
            ["https://www.youtube.com/watch?v=rws_ieEZVao"]  # Kurzgesagt
        ],
        inputs=[url_input],
        outputs=[output],
        fn=gradio_summarize_wrapper,
        label="📋 Try These Examples:",
        cache_examples=False # Set to True for production if inputs/outputs are consistent
    )

app = gr.mount_gradio_app(app, gradio_interface, path="/")
