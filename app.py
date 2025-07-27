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
        # SPEED OPTIMIZATION: Try a faster model first
        model_options = [
            "facebook/bart-large-cnn",          # Usually faster than distilbart
            "sshleifer/distilbart-cnn-12-6",    # Original choice
            "google/pegasus-xsum",              # Alternative fast option
        ]
        
        model_name = os.getenv("MODEL_NAME", model_options[0])
        
        # If model_name is not in our list, try it anyway
        if model_name not in model_options:
            model_options.insert(0, model_name)
        
        for model in model_options:
            try:
                logger.info(f"Trying to load model: {model}")
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    device=-1,
                    # Speed optimizations
                    torch_dtype="auto",
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded model: {model}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
                if model == model_options[-1]:  # Last model failed
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
    """Parse VTT subtitle content and extract text."""
    lines = vtt_content.split('\n')
    text_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip VTT headers, timestamps, and empty lines
        if (line.startswith('WEBVTT') or 
            line.startswith('Kind:') or 
            line.startswith('Language:') or
            '-->' in line or 
            line.isdigit() or 
            not line):
            continue
        
        # Remove HTML tags and timing info
        line = re.sub(r'<[^>]+>', '', line)
        line = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', line)
        
        if line:
            text_lines.append(line)
    
    return ' '.join(text_lines)

def get_transcript_with_ytdlp(youtube_url: str):
    """Get transcript using yt-dlp - much more reliable than youtube-transcript-api."""
    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format provided.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Use yt-dlp to download subtitles only
            cmd = [
                'yt-dlp',
                '--write-auto-subs',  # Download auto-generated subtitles
                '--write-subs',       # Download manual subtitles if available
                '--sub-langs', 'en,en-US,en-GB',  # Prefer English
                '--sub-format', 'vtt',  # VTT format is easier to parse
                '--skip-download',    # Don't download the video
                '--output', f'{temp_dir}/%(title)s.%(ext)s',
                youtube_url
            ]
            
            logger.info(f"Running yt-dlp command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"yt-dlp failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                raise RuntimeError(f"Failed to fetch subtitles: {result.stderr}")
            
            # Find the subtitle files
            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            
            if not vtt_files:
                raise RuntimeError("No subtitle files were downloaded. This video may not have captions available.")
            
            # Prefer manual subtitles over auto-generated ones
            manual_subs = [f for f in vtt_files if '.en.' in f and 'auto' not in f]
            auto_subs = [f for f in vtt_files if '.en.' in f and 'auto' in f]
            
            subtitle_file = manual_subs[0] if manual_subs else auto_subs[0] if auto_subs else vtt_files[0]
            
            logger.info(f"Using subtitle file: {subtitle_file}")
            
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            
            transcript_text = parse_vtt_content(vtt_content)
            
            if not transcript_text or len(transcript_text.split()) < 10:
                raise RuntimeError("The extracted transcript is too short or empty.")
            
            return transcript_text
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Transcript fetch timed out. Please try again.")
        except Exception as e:
            logger.error(f"Error in get_transcript_with_ytdlp: {e}", exc_info=True)
            raise

def create_extractive_summary(text, num_sentences=6):
    """Create a fast extractive summary by selecting key sentences."""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Simple scoring: prefer sentences with common keywords
    important_words = ['important', 'key', 'main', 'first', 'second', 'finally', 'conclusion', 'result', 'because', 'therefore', 'however', 'moreover', 'furthermore']
    
    scored_sentences = []
    for sentence in sentences:
        score = 0
        words = sentence.lower().split()
        # Score based on length (not too short, not too long)
        if 10 <= len(words) <= 30:
            score += 2
        # Score based on important words
        score += sum(1 for word in important_words if word in sentence.lower())
        # Score based on position (beginning and end are often important)
        pos = sentences.index(sentence)
        if pos < len(sentences) * 0.2 or pos > len(sentences) * 0.8:
            score += 1
        
        scored_sentences.append((score, sentence))
    
    # Select top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    selected = [sentence for score, sentence in scored_sentences[:num_sentences]]
    
    # Reorder selected sentences by their original position
    ordered_summary = []
    for sentence in sentences:
        if sentence in selected:
            ordered_summary.append(sentence)
    
    return '. '.join(ordered_summary) + '.'

def process_and_summarize(youtube_url: str):
    if is_model_loading:
        raise RuntimeError("The AI model is still loading. Please try again in a few minutes.")
    if summarizer is None:
        raise RuntimeError(f"Model is not available. Load error: {model_load_error or 'Unknown reason.'}")

    try:
        # Get transcript using yt-dlp
        logger.info("Fetching transcript using yt-dlp...")
        full_transcript = get_transcript_with_ytdlp(youtube_url)
        logger.info(f"Successfully fetched transcript with {len(full_transcript.split())} words")
        
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}", exc_info=True)
        raise RuntimeError(f"Unable to fetch transcript: {str(e)}")

    if len(full_transcript.split()) < 50:
        raise ValueError("Transcript is too short to generate a meaningful summary.")

    # SPEED OPTIMIZATION 1: Reduce transcript size intelligently
    words = full_transcript.split()
    
    # If transcript is very long, take strategic samples instead of processing everything
    if len(words) > 2000:
        # Take beginning, middle, and end sections
        beginning = words[:600]
        middle_start = len(words) // 2 - 300
        middle = words[middle_start:middle_start + 600]
        end = words[-600:]
        words = beginning + middle + end
        logger.info(f"Large transcript detected. Using strategic sampling: {len(words)} words")
    
    sampled_text = " ".join(words)
    
    # EMERGENCY SPEED OPTION: If AI is too slow, use extractive summary
    try:
        # Set a practical timeout approach
        logger.info("Starting AI summarization with speed optimizations...")
        
        # SPEED OPTIMIZATION 2: Use smaller, faster chunks
        chunk_size = 300  # Even smaller chunks
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        # SPEED OPTIMIZATION 3: Process only 1-2 chunks maximum
        max_chunks = min(2, len(chunks))
        
        summaries = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            logger.info(f"AI Summarizing chunk {i+1}/{max_chunks}")
            
            # SPEED OPTIMIZATION 4: Fastest possible parameters
            summary = summarizer(
                chunk, 
                max_length=80,   # Even shorter
                min_length=15,   # Even shorter minimum
                do_sample=False,
                truncation=True,
                num_beams=1,     # No beam search for maximum speed
                early_stopping=True,
                repetition_penalty=1.1
            )[0]['summary_text']
            
            summaries.append(summary)
            logger.info(f"AI chunk {i+1} completed successfully")
        
        if summaries:
            final_summary = "\n\n".join([f"• {s}" for s in summaries])
            total_words_processed = len(words)
            note = f"\n\n*(AI-generated summary from {total_words_processed} words using advanced language model.)*"
            return final_summary + note
            
    except Exception as e:
        logger.warning(f"AI summarization failed or too slow: {e}")
        logger.info("Falling back to fast extractive summarization...")
        
        # FALLBACK: Ultra-fast extractive summary
        extractive_summary = create_extractive_summary(sampled_text, num_sentences=8)
        note = f"\n\n*(Fast extractive summary from {len(words)} words - AI model was unavailable or too slow.)*"
        return f"• {extractive_summary}" + note

# --- API Endpoint ---
@app.post("/api/summarize/")
async def api_summarize(request: SummarizeRequest):
    try:
        summary = process_and_summarize(request.youtube_url)
        return JSONResponse(content={"summary": summary})
    except (ValueError, RuntimeError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Gradio Interface ---
with gr.Blocks(title="YouTube Video Summarizer", theme=gr.themes.Soft()) as gradio_interface:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("Enter a YouTube URL to get an AI-generated summary using **yt-dlp** (more reliable than other methods).")
    
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
