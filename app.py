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
    """Loads the model in a background thread, using a fast default."""
    global summarizer, model_load_error, is_model_loading
    if summarizer or is_model_loading:
        return
    is_model_loading = True
    logger.info("Background thread: Starting model loading...")
    
    try:
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
    thread = threading.Thread(target=load_model_proc)
    thread.start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- Core Logic Functions ---
def get_transcript_with_ytdlp(youtube_url: str):
    """More reliable transcript fetching using yt-dlp."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            'yt-dlp', '--write-auto-subs', '--sub-langs', 'en.*',
            '--write-subs', '--sub-format', 'vtt', '--skip-download',
            '--no-playlist', '--output', f'{temp_dir}/%(id)s.%(ext)s', youtube_url
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        if proc.returncode != 0:
            error_message = proc.stderr.strip()
            logger.error(f"yt-dlp failed: {error_message}")
            if "copyright" in error_message.lower(): raise RuntimeError("Failed to fetch transcript due to a copyright claim.")
            if "private video" in error_message.lower(): raise RuntimeError("Failed to fetch transcript. The video is private.")
            raise RuntimeError("yt-dlp failed. The video may be unavailable or have no English captions.")
        
        vtt_files = glob.glob(f"{temp_dir}/*.vtt")
        if not vtt_files: raise RuntimeError("No English subtitle file was downloaded.")
        
        with open(vtt_files[0], 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        text_lines = [line for line in lines if line and '-->' not in line and 'WEBVTT' not in line]
        transcript = ' '.join(text_lines)
        
        if len(transcript.split()) < 50: raise ValueError("Transcript is too short to provide a quality summary.")
        return transcript

def generate_ai_overview(intro_text, conclusion_text, progress):
    """Generates a high-level overview by summarizing the beginning and end."""
    if not summarizer: return "AI model not loaded."
    progress(0.5, desc="AI is generating the overview...")
    combined_text = f"Introduction: {intro_text} [...] Conclusion: {conclusion_text}"
    
    try:
        # === THE FIX: Added truncation=True ===
        overview = summarizer(combined_text, max_length=100, min_length=25, do_sample=False, truncation=True)[0]['summary_text']
        return overview
    except Exception as e:
        logger.error(f"Error during AI overview generation: {e}")
        return "Could not generate AI overview for this section."

def generate_ai_recap(middle_text, progress):
    """Generates a detailed recap by summarizing the core content of the video."""
    if not summarizer: return ["AI model not loaded."]
    progress(0.75, desc="AI is recapping key moments...")
    words = middle_text.split()
    midpoint = len(words) // 2
    chunk1 = " ".join(words[:midpoint])
    chunk2 = " ".join(words[midpoint:])
    
    recap_points = []
    try:
        # === THE FIX: Added truncation=True to both calls ===
        recap1 = summarizer(chunk1, max_length=120, min_length=20, do_sample=False, truncation=True)[0]['summary_text']
        recap_points.append(recap1)
        
        progress(0.9, desc="AI is recapping final moments...")
        recap2 = summarizer(chunk2, max_length=120, min_length=20, do_sample=False, truncation=True)[0]['summary_text']
        recap_points.append(recap2)
    except Exception as e:
        logger.error(f"Error during AI recap generation: {e}")
        recap_points.append("Could not generate AI recap for a section.")
    return recap_points

# --- Gradio Interface ---
with gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Get a high-quality, structured AI summary of any YouTube video with captions.")
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
        try:
            progress(0, desc="Contacting YouTube...")
            if not youtube_url: raise ValueError("Please enter a YouTube URL.")
            progress(0.1, desc="Fetching Transcript with yt-dlp...")
            transcript = get_transcript_with_ytdlp(youtube_url)
            progress(0.4, desc="Analyzing transcript structure...")
            words = transcript.split()
            word_count = len(words)
            intro_word_count = min(300, int(word_count * 0.2))
            conclusion_word_count = min(350, int(word_count * 0.25))
            intro_text = " ".join(words[:intro_word_count])
            conclusion_text = " ".join(words[-conclusion_word_count:])
            middle_text = " ".join(words[intro_word_count:-conclusion_word_count])
            
            overview = generate_ai_overview(intro_text, conclusion_text, progress)
            recap_points = generate_ai_recap(middle_text, progress)
            
            progress(1.0, desc="Done!")
            final_output = f"## 📖 AI Overview\n\n*{overview}*\n\n---\n\n## 🎬 AI Recap of Key Moments\n\n"
            for point in recap_points:
                final_output += f"• **{point}**\n\n"
            final_output += f"\n---\n*Analysis based on a transcript of {word_count:,} words.*"
            yield final_output
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            yield f"❌ **Error:** {str(e)}"

    submit_btn.click(fn=gradio_summarize_wrapper, inputs=[url_input], outputs=[output])
    gradio_interface.load(get_status, None, status_display, every=3)
    gr.Examples(
        examples=[["https://www.youtube.com/watch?v=jNQXAC9IVRw"], ["https://www.youtube.com/watch?v=9bZkp7q19f0"]],
        inputs=[url_input], outputs=[output], fn=gradio_summarize_wrapper
    )

app = gr.mount_gradio_app(app, gradio_interface, path="/")
