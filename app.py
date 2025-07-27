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
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
        # SPEED OPTIMIZATION: Try faster, more reliable models
        model_options = [
            "sshleifer/distilbart-cnn-6-6",     # Smaller, faster version
            "sshleifer/distilbart-cnn-12-6",    # Original choice
            "facebook/bart-large-cnn",          # Alternative
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
                    # Optimizations for speed and stability
                    framework="pt",
                    return_tensors="pt",
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

def create_extractive_summary(text, num_sentences=10):
    """Create a fast extractive summary by selecting key sentences with better formatting."""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Enhanced scoring: prefer sentences with common keywords and better structure
    important_words = [
        'important', 'key', 'main', 'first', 'second', 'third', 'finally', 'conclusion', 
        'result', 'because', 'therefore', 'however', 'moreover', 'furthermore', 'essentially',
        'basically', 'specifically', 'particularly', 'especially', 'according', 'studies',
        'research', 'found', 'shows', 'indicates', 'suggests', 'explains', 'means',
        'example', 'instance', 'such as', 'including', 'problem', 'solution', 'issue',
        'benefit', 'advantage', 'disadvantage', 'effect', 'impact', 'influence'
    ]
    
    scored_sentences = []
    for sentence in sentences:
        score = 0
        words = sentence.lower().split()
        
        # Score based on length (prefer medium-length sentences)
        if 8 <= len(words) <= 25:
            score += 3
        elif 25 < len(words) <= 35:
            score += 2
        elif len(words) > 35:
            score += 1
        
        # Score based on important words
        for important_word in important_words:
            if important_word in sentence.lower():
                score += 2
        
        # Score based on position (beginning and end are often important)
        pos = sentences.index(sentence)
        if pos < len(sentences) * 0.15:  # First 15%
            score += 3
        elif pos > len(sentences) * 0.85:  # Last 15%
            score += 2
        elif 0.4 <= pos/len(sentences) <= 0.6:  # Middle section
            score += 1
        
        # Boost sentences with numbers or statistics
        if any(char.isdigit() for char in sentence):
            score += 1
        
        # Boost sentences with quotes or direct speech
        if '"' in sentence or "'" in sentence:
            score += 1
        
        scored_sentences.append((score, sentence))
    
    # Select top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    selected = [sentence for score, sentence in scored_sentences[:num_sentences]]
    
    # Reorder selected sentences by their original position for coherence
    ordered_summary = []
    for sentence in sentences:
        if sentence in selected:
            ordered_summary.append(sentence)
    
    return '. '.join(ordered_summary) + '.'

def format_ai_summary(summaries, total_words):
    """Format AI-generated summaries with better styling and structure."""
    formatted_summary = "## 🤖 **AI-Generated Summary**\n\n"
    
    if len(summaries) == 1:
        formatted_summary += f"**Key Points:**\n\n• {summaries[0]}\n\n"
    else:
        for i, summary in enumerate(summaries, 1):
            if i == 1:
                formatted_summary += f"**Part {i} - Opening/Introduction:**\n• {summary}\n\n"
            elif i == len(summaries):
                formatted_summary += f"**Part {i} - Conclusion/Key Takeaways:**\n• {summary}\n\n"
            else:
                formatted_summary += f"**Part {i} - Main Content:**\n• {summary}\n\n"
    
    formatted_summary += f"*📊 Analysis: Processed {total_words:,} words using advanced AI language models*"
    return formatted_summary

def format_extractive_summary(summary, total_words):
    """Format extractive summary with better styling."""
    # Split into logical sections
    sentences = [s.strip() + '.' for s in summary.split('.') if s.strip()]
    
    formatted_summary = "## 📝 **Quick Summary**\n\n"
    formatted_summary += "**Main Points:**\n\n"
    
    for i, sentence in enumerate(sentences[:6], 1):  # Limit to 6 main points
        if sentence.strip():
            formatted_summary += f"**{i}.** {sentence}\n\n"
    
    if len(sentences) > 6:
        formatted_summary += "**Additional Context:**\n\n"
        for sentence in sentences[6:]:
            if sentence.strip():
                formatted_summary += f"• {sentence}\n\n"
    
    formatted_summary += f"*⚡ Fast Summary: Extracted key information from {total_words:,} words*"
    return formatted_summary

def summarize_with_timeout(summarizer, chunk, timeout_seconds=45):
    """Summarize a chunk with a timeout to prevent hanging."""
    def summarize_chunk():
        return summarizer(
            chunk, 
            max_length=120,  # Increased for better summaries
            min_length=25,   # Increased minimum
            do_sample=False,
            truncation=True,
            pad_token_id=summarizer.tokenizer.eos_token_id,
        )[0]['summary_text']
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(summarize_chunk)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            logger.warning(f"Summarization timed out after {timeout_seconds} seconds")
            return None

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

    # Enhanced text processing for better summaries
    words = full_transcript.split()
    original_word_count = len(words)
    
    # Smart sampling strategy based on video length
    if len(words) > 3000:
        # For very long videos: take more strategic samples
        beginning = words[:800]
        quarter = words[len(words)//4:len(words)//4 + 600]
        middle = words[len(words)//2 - 400:len(words)//2 + 400]  
        three_quarter = words[3*len(words)//4:3*len(words)//4 + 600]
        end = words[-800:]
        words = beginning + quarter + middle + three_quarter + end
        logger.info(f"Very long transcript detected. Using comprehensive sampling: {len(words)} words from {original_word_count}")
    elif len(words) > 1500:
        # For medium videos: standard sampling
        beginning = words[:600]
        middle = words[len(words)//2 - 400:len(words)//2 + 400]
        end = words[-600:]
        words = beginning + middle + end
        logger.info(f"Medium transcript detected. Using strategic sampling: {len(words)} words from {original_word_count}")
    
    sampled_text = " ".join(words)
    
    # Try AI summarization with improved parameters
    try:
        logger.info("Starting enhanced AI summarization...")
        
        # Dynamic chunk sizing based on content length
        if len(words) > 2000:
            chunk_size = 500
            max_chunks = 3
        elif len(words) > 1000:
            chunk_size = 400  
            max_chunks = 3
        else:
            chunk_size = 300
            max_chunks = 2
        
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        chunks_to_process = min(max_chunks, len(chunks))
        
        summaries = []
        for i, chunk in enumerate(chunks[:chunks_to_process]):
            logger.info(f"AI Summarizing section {i+1}/{chunks_to_process} with 45-second timeout")
            
            # Use timeout-protected summarization with better parameters
            summary = summarize_with_timeout(summarizer, chunk, timeout_seconds=45)
            
            if summary is None:
                logger.warning(f"Section {i+1} timed out, switching to extractive method")
                break
            
            summaries.append(summary)
            logger.info(f"AI section {i+1} completed successfully")
        
        # If we got AI summaries, format them nicely
        if summaries:
            return format_ai_summary(summaries, len(words))
            
    except Exception as e:
        logger.warning(f"AI summarization failed: {e}")
    
    # FALLBACK: Enhanced extractive summary
    logger.info("Using enhanced extractive summarization...")
    extractive_summary = create_extractive_summary(sampled_text, num_sentences=12)
    return format_extractive_summary(extractive_summary, len(words))

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
    gr.Markdown("**Get comprehensive AI-generated summaries** of YouTube videos using **yt-dlp** for reliable transcript extraction.")
    
    status_display = gr.Markdown("Model status: Unknown")

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube Video URL", 
            placeholder="https://www.youtube.com/watch?v=...", 
            lines=1, 
            scale=3
        )
        submit_btn = gr.Button("🚀 Summarize", variant="primary", scale=1)
    
    output = gr.Markdown(label="Summary", height=400)

    def get_status():
        if summarizer:
            return "✅ **Model Status:** Loaded and ready for AI summarization"
        if is_model_loading:
            return "⏳ **Model Status:** Loading in the background... (this may take several minutes)"
        if model_load_error:
            return f"❌ **Model Status:** Failed to load - {model_load_error}"
        return "⏳ **Model Status:** Initializing..."

    def gradio_summarize_wrapper(youtube_url: str):
        try:
            summary = process_and_summarize(youtube_url)
            return summary  # No need for ✅ prefix since formatting is handled internally
        except Exception as e:
            return f"❌ **Error:** {e}"

    submit_btn.click(fn=gradio_summarize_wrapper, inputs=[url_input], outputs=[output])
    
    gradio_interface.load(get_status, None, status_display, every=3)
    
    gr.Examples(
        examples=[
            ["https://www.youtube.com/watch?v=jNQXAC9IVRw"], 
            ["https://www.youtube.com/watch?v=9bZkp7q19f0"],
            ["https://www.youtube.com/watch?v=rws_ieEZVao"]
        ],
        inputs=[url_input], 
        outputs=[output], 
        fn=gradio_summarize_wrapper,
        label="📋 **Try These Examples:**"
    )

app = gr.mount_gradio_app(app, gradio_interface, path="/")
