import gradio as gr
from transformers import pipeline
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re
import logging
import threading
import subprocess
import tempfile
import glob

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global state variables ---
summarizer = None
model_load_error = None
is_model_loading = False

# --- Model Loading Logic ---
def load_model_proc():
    """Loads the model in a background thread, trying faster models first."""
    global summarizer, model_load_error, is_model_loading
    if summarizer or is_model_loading:
        return
    is_model_loading = True
    logger.info("Background thread: Starting model loading...")
    
    try:
        # Use a smaller, faster model by default for better performance on CPU
        model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-6-6")
        summarizer = pipeline("summarization", model=model_name, device=-1)
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
    thread = threading.Thread(target=load_model_proc)
    thread.start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- Core Logic Functions ---
def get_video_id(url: str):
    if not url: return None
    patterns = [r'v=([a-zA-Z0-9_-]{11})', r'be\/([a-zA-Z0-9_-]{11})']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript_with_ytdlp(youtube_url: str):
    """More reliable transcript fetching using yt-dlp."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            'yt-dlp',
            '--write-auto-subs', '--sub-langs', 'en.*',
            '--write-subs',
            '--sub-format', 'vtt',
            '--skip-download',
            '--no-playlist',
            '--output', f'{temp_dir}/%(id)s.%(ext)s',
            youtube_url
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        if proc.returncode != 0:
            error_message = proc.stderr.strip()
            logger.error(f"yt-dlp failed: {error_message}")
            if "copyright" in error_message.lower():
                raise RuntimeError("Failed to fetch transcript due to a copyright claim on the video.")
            if "private video" in error_message.lower():
                 raise RuntimeError("Failed to fetch transcript. The video is private.")
            raise RuntimeError("yt-dlp failed. The video may be unavailable or have no English captions.")
        
        vtt_files = glob.glob(f"{temp_dir}/*.vtt")
        if not vtt_files:
            raise RuntimeError("No English subtitle file was downloaded.")
        
        with open(vtt_files[0], 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        # Simple but effective VTT parsing
        text_lines = [line for line in lines if line and '-->' not in line and 'WEBVTT' not in line]
        transcript = ' '.join(text_lines)
        
        if len(transcript.split()) < 30:
            raise ValueError("Transcript is too short to summarize.")
            
        return transcript

def create_extractive_summary(text, num_sentences=8):
    """Creates a fast summary by picking the most important sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences: return "Could not generate a quick summary."
    
    # Very simple scoring: longer sentences are more important
    scored_sentences = sorted([(len(s.split()), s) for s in sentences if len(s.split()) > 5], reverse=True)
    
    top_sentences = [s for _, s in scored_sentences[:num_sentences]]
    
    # Reorder them to appear as they did in the original text
    summary_sentences = [s for s in sentences if s in top_sentences]
    
    return ' '.join(summary_sentences)

def generate_ai_summary(text, progress):
    """Generates the high-quality abstractive summary."""
    if not summarizer:
        return "\n\n**Note:** AI model is not loaded, so a deep summary could not be generated."

    words = text.split()
    # For performance, only use the first ~1500 words for the AI summary on a CPU
    text_to_summarize = " ".join(words[:1500])
    
    num_chunks = min(3, (len(text_to_summarize.split()) // 400) + 1)
    chunk_size = len(text_to_summarize.split()) // num_chunks
    chunks = [" ".join(text_to_summarize.split()[i:i+chunk_size]) for i in range(0, len(text_to_summarize.split()), chunk_size)]

    summaries = []
    for i, chunk in enumerate(chunks[:3]): # Max 3 chunks
        progress(0.6 + (i / len(chunks) * 0.4), desc=f"AI is processing section {i+1}/{len(chunks)}...")
        try:
            summary = summarizer(chunk, max_length=120, min_length=25, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error summarizing chunk {i}: {e}")
            summaries.append(f"(AI processing failed for this section.)")
            
    return "\n".join(f"• {s.strip()}" for s in summaries)


# --- Gradio Interface ---
with gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Get a near-instant quick summary, followed by a deeper AI-generated summary.")
    
    status_display = gr.Markdown("Model status: Unknown")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", scale=3)
        submit_btn = gr.Button("🚀 Summarize", variant="primary")
    
    output = gr.Markdown(label="Summary")

    def get_status():
        if summarizer: return "✅ **Model Status:** Ready for AI summarization."
        if is_model_loading: return "⏳ **Model Status:** Loading in background... (may take several minutes)"
        if model_load_error: return f"❌ **Model Status:** Failed to load - {model_load_error}"
        return "⏳ **Model Status:** Initializing..."

    def gradio_summarize_wrapper(youtube_url: str, progress=gr.Progress(track_tqdm=True)):
        """Orchestrates the new two-stage summary process with a progress bar."""
        try:
            # === STAGE 1: Fast Transcript & Extractive Summary ===
            progress(0, desc="Contacting YouTube...")
            if not youtube_url:
                raise ValueError("Please enter a YouTube URL.")
                
            progress(0.1, desc="Fetching Transcript with yt-dlp...")
            transcript = get_transcript_with_ytdlp(youtube_url)
            
            progress(0.4, desc="Generating Quick Summary...")
            quick_summary = create_extractive_summary(transcript)
            
            # YIELD the first, fast result to the UI
            yield f"## 📝 **Quick Summary (Instant)**\n\n{quick_summary}"

            # === STAGE 2: Slow, High-Quality AI Summary ===
            progress(0.6, desc="Starting Deep AI Analysis...")
            ai_summary_points = generate_ai_summary(transcript, progress)
            
            # YIELD the final, combined result
            progress(1.0, desc="Done!")
            yield f"## 📝 **Quick Summary**\n\n{quick_summary}\n\n---\n\n## 🤖 **Deep AI Summary**\n\n{ai_summary_points}"

        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            yield f"❌ **Error:** {str(e)}"

    submit_btn.click(
        fn=gradio_summarize_wrapper, 
        inputs=[url_input], 
        outputs=[output]
    )
    
    gradio_interface.load(get_status, None, status_display, every=3)
    
    gr.Examples(
        examples=[["https://www.youtube.com/watch?v=jNQXAC9IVRw"], ["https://www.youtube.com/watch?v=9bZkp7q19f0"]],
        inputs=[url_input], outputs=[output], fn=gradio_summarize_wrapper
    )

app = gr.mount_gradio_app(app, gradio_interface, path="/")
