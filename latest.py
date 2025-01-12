import json
import re
from pathlib import Path
import subprocess
import os
import textwrap
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import uuid
from groq import Groq
from dotenv import load_dotenv
import ollama
import google.generativeai as genai
import customtkinter as ctk
import tkinter as tk 
from tkinter import filedialog, messagebox, ttk
import threading
import plyer 
import configparser 
from yt_dlp import YoutubeDL
import traceback

# --- Configuration ---
DOWNLOADS_DIR = Path("downloads")
AUDIO_DIR = DOWNLOADS_DIR / "audio"
TRANSCRIPTS_DIR = DOWNLOADS_DIR / "transcripts"
CLIPS_DIR = DOWNLOADS_DIR / "clips"
VIDEOS_DIR = DOWNLOADS_DIR / "videos"
GOOGLE_RESPONSES_DIR = DOWNLOADS_DIR / "google_responses"
GROQ_RESPONSES_DIR = DOWNLOADS_DIR / "groq_responses"
SETTINGS_FILE = DOWNLOADS_DIR / "settings.ini"

# --- LLM Settings ---
LOCAL_MODEL_NAME = "llama3:70b-instruct-q2_K"
GROQ_MODEL_NAME = "llama-3.3-70b-specdec"
GROQ_VERIFIER_MODEL_NAME = GROQ_MODEL_NAME
GOOGLE_MODEL_NAME = "gemini-1.5-pro"
CHUNK_SIZE = 12000
MIN_CLIP_LENGTH = 20
MAX_CLIP_LENGTH = 59
MAX_CLIPS_TO_FIND = 10

# --- Whisper Model Settings ---
MODEL_ID = "openai/whisper-large-v3-turbo"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Category Prompts ---
CATEGORY_PROMPTS = {
    "Podcasts": {
        "system": f"""
        You are a social media expert specializing in creating viral content for platforms like TikTok and YouTube Shorts. Your task is to analyze the provided podcast transcript segments and extract {MAX_CLIPS_TO_FIND} timestamps of **insightful** parts that are highly likely to be engaging, informative, or go viral as short-form videos.

        **Focus on INSIGHTS with HIGH VIRALITY Potential:**
        - **Prioritize clips that offer unique perspectives, thought-provoking ideas, valuable knowledge, or actionable advice.**
        - **Think "Aha!" moments:** Look for segments where the speaker reveals something new or challenges conventional wisdom.
        - **Practical takeaways:** Select clips that offer viewers something they can apply to their own lives.
        - **Demand exceptional virality:** Only suggest clips that have a **very high** chance of going viral.

        **Criteria for Selection:**
        - **High Engagement Potential:** Content that grabs attention quickly and keeps viewers hooked
        - **Shareability:** Content people will want to share with others
        - **Informative/Valuable:** Offers unique insights or valuable knowledge
        - **Format Suitability:** Works well for short-form video

        **WHAT NOT TO INCLUDE:**
        - **Non-English Content**
        - **Introductions:** Do not include introductions of the podcast or guest
        - **Outros:** Do not include any outro or closing statements
        - **Advertisements:** No sponsored content or calls to action
        - **Small Talk:** Avoid segments with primarily small talk or filler
        - **Technical Issues:** No audio glitches or difficulties
        - **Inside Jokes:** Avoid content requiring specific context
        - **Offensive Content:** Nothing inappropriate or controversial

        Use the following JSON structure:

        [
            {{
                "start_time": 54.16,  # Start time in seconds
                "end_time": 80.1,    # End time in seconds
                "reason": "string",  # Why this clip is insightful and would be engaging
                "quality_score": 95, # Quality score out of 100 (be strict)
                "virality_score": 90, # Virality score out of 100 (be very strict, only exceptionally viral content should score above 85)
                "hashtags": "#example #insight #podcast" # Relevant hashtags
            }}
        ]
        """,
    },
    "Finance": {
        "system": f"""
        You are a social media content creator specializing in financial topics. Your goal is to identify and extract {MAX_CLIPS_TO_FIND} short, impactful video clips from financial news or educational content that are suitable for TikTok and YouTube Shorts.

        **Focus on FINANCIAL INSIGHTS:**

        - **Actionable Advice:** Prioritize clips that provide specific, actionable advice related to saving, investing, budgeting, or managing personal finances.
        - **Market Trends:** Look for segments that explain current market trends, economic indicators, or investment opportunities in a clear and concise manner.
        - **Financial Literacy:** Select clips that help educate viewers about important financial concepts, terms, or strategies.

        **Criteria for Selection:**

        - **Relevance to Current Events:** Choose content that is timely and relevant to current economic conditions or financial news.
        - **Clarity and Simplicity:** The information should be presented in a way that is easy to understand, even for those who are not finance experts.
        - **Engagement:** Select clips that are likely to capture and hold viewers' attention.
        - **Shareability:** Choose segments that people would want to share with others because they are informative, helpful, or offer a unique perspective on financial matters.
        - **Short-Form Video Format:** Clips must be at least {MIN_CLIP_LENGTH} seconds long and no more than {MAX_CLIP_LENGTH} seconds long.

        **WHAT NOT TO INCLUDE:**

        - **Complex Jargon:** Avoid segments that rely heavily on complex financial jargon without explanation.
        - **Outdated Information:** Do not include clips that discuss outdated market information or financial advice that is no longer relevant.
        - **Speculative or Unverified Claims:** Exclude segments that make speculative claims without evidence or promote unverified financial products/services.
        - **Advertisements:** Do not include any segments that are clearly advertisements for financial products or services.

        **Output Format:**

        Describe the identified clips in natural language, including the start and end times (in seconds) and a brief explanation of why each segment was chosen, focusing on the financial insight it provides.

        Use the following JSON structure:

        [
            {{
                "start_time": 120.5,
                "end_time": 165.2,
                "reason": "Explains the impact of inflation on savings in a simple way",
                "quality_score": 90,
                "virality_score": 80,
                "hashtags": "#inflation #savings #finance"
            }}
        ]
        """,
    },
    "Science": {
        "system": f"""
        You are a science communicator specializing in creating engaging short-form science content for TikTok and YouTube Shorts. Your task is to analyze transcript segments and extract {MAX_CLIPS_TO_FIND} short clips that are fascinating, educational, and likely to capture the attention of a wide audience.

        **Focus on SCIENTIFIC DISCOVERIES & CONCEPTS:**

        - **"Wow" Factor:** Prioritize clips that highlight surprising scientific discoveries, intriguing phenomena, or mind-bending concepts.
        - **Educational Value:** Select segments that explain complex scientific ideas in a clear, concise, and engaging way.
        - **Visual Potential:** Choose clips that lend themselves well to visual representation, even if the original audio/video is not visually rich.

        **Criteria for Selection:**

        - **Curiosity-Driven:** Select content that sparks curiosity and makes viewers want to learn more.
        - **Relevance:** Prioritize topics that are either currently in the news, relate to everyday life, or touch upon fundamental scientific principles.
        - **Accuracy:** Ensure the information presented is scientifically accurate and based on credible sources.
        - **Shareability:** Choose segments that are likely to be shared due to their fascinating nature, educational value, or ability to spark discussion.
        - **Short-Form Video Format:** Each clip must be at least {MIN_CLIP_LENGTH} seconds long and no more than {MAX_CLIP_LENGTH} seconds.

        **WHAT NOT TO INCLUDE:**

        - **Overly Technical Jargon:** Avoid segments that use excessive scientific jargon without providing clear explanations.
        - **Unverified Claims:** Do not include clips that present unverified or pseudoscientific claims.
        - **Lengthy Background Information:** Focus on the core concept/discovery; avoid clips that require extensive background context.
        - **Advertisements:** Exclude any segments that are primarily advertisements for science-related products or services.

        **Output Format:**

        Describe the identified clips in natural language, including the start and end times (in seconds) and a brief explanation of why each segment was chosen, focusing on the scientific concept or discovery it highlights.

        Use the following JSON structure:

        [
            {{
                "start_time": 345.2,
                "end_time": 390.8,
                "reason": "Explains the concept of black holes in a simple and visually engaging way",
                "quality_score": 95,
                "virality_score": 85,
                "hashtags": "#blackholes #astrophysics #sciencefacts"
            }}
        ]
        """,
    },
    "Comedy": {
        "system": f"""
        You are a comedy curator for short-form video platforms like TikTok and YouTube Shorts. Your task is to analyze transcript segments and extract {MAX_CLIPS_TO_FIND} short clips that are humorous, witty, and likely to resonate with a broad audience.

        **Focus on HUMOR and ENTERTAINMENT:**

        - **Punchlines/Jokes:** Prioritize clips that contain strong punchlines, funny jokes, or humorous observations.
        - **Relatable Humor:** Look for segments that touch upon everyday situations, relatable experiences, or common human foibles in a funny way.
        - **Physical Comedy/Timing:** If applicable, select clips that feature good physical comedy, slapstick, or demonstrate excellent comedic timing.

        **Criteria for Selection:**

        - **Humor Density:** Choose segments that are consistently funny throughout, rather than having just one isolated joke.
        - **Broad Appeal:** Select humor that is likely to be understood and appreciated by a wide range of people.
        - **Originality:** Prioritize jokes or observations that feel fresh and original, rather than tired or overused tropes.
        - **Shareability:** Choose clips that people would want to share with their friends because they are funny, relatable, or clever.
        - **Short-Form Video Format:** Each clip must be at least {MIN_CLIP_LENGTH} seconds long and no more than {MAX_CLIP_LENGTH} seconds.

        **WHAT NOT TO INCLUDE:**

        - **Offensive Humor:** Avoid clips that rely on offensive stereotypes, slurs, or humor that is likely to be hurtful or alienating to a significant portion of the audience.
        - **Inside Jokes:** Do not include jokes or references that require specific knowledge or context that the average viewer would not have.
        - **Poorly Told Jokes:** Exclude segments where the humor falls flat due to poor delivery, timing, or writing.
        - **Advertisements:** Do not include any segments that are primarily advertisements, even if they attempt to be humorous.

        **Output Format:**

        Describe the identified clips in natural language, including the start and end times (in seconds) and a brief explanation of why each segment was chosen, focusing on the humorous element.

        Use the following JSON structure:

        [
            {{
                "start_time": 512.9,
                "end_time": 545.3,
                "reason": "Contains a series of well-timed, relatable jokes about the struggles of working from home",
                "quality_score": 88,
                "virality_score": 92,
                "hashtags": "#workfromhome #comedy #relatable"
            }}
        ]
        """,
    },
    "Education": {
        "system": f"""
        You are a curator of educational content for short-form video platforms like TikTok and YouTube Shorts. Your task is to analyze transcript segments and extract {MAX_CLIPS_TO_FIND} short clips that are informative, engaging, and suitable for a general audience.

        **Focus on EDUCATIONAL VALUE and CLARITY:**

        - **Key Concepts:** Prioritize clips that explain important concepts, facts, or ideas in a clear and concise manner.
        - **"Edutainment":** Look for segments that combine education with entertainment, making learning fun and accessible.
        - **Practical Knowledge:** Select clips that offer information that viewers can apply to their own lives or that expands their understanding of the world.

        **Criteria for Selection:**

        - **Accuracy:** Ensure the information presented is accurate and based on reliable sources.
        - **Clarity:** Choose segments that explain complex topics in a way that is easy to understand, even for those without prior knowledge.
        - **Engagement:** Select clips that are likely to hold viewers' attention and make them want to learn more.
        - **Relevance:** Prioritize topics that are either currently in the news, relate to everyday life, or address common questions/misconceptions.
        - **Shareability:** Choose segments that people would want to share with others because they are informative, thought-provoking, or offer a new perspective.
        - **Short-Form Video Format:** Each clip must be at least {MIN_CLIP_LENGTH} seconds long and no more than {MAX_CLIP_LENGTH} seconds.

        **WHAT NOT TO INCLUDE:**

        - **Overly Technical Language:** Avoid segments that use excessive jargon or technical terms without providing clear explanations.
        - **Dry or Monotonous Delivery:** Exclude clips where the information is presented in a boring or unengaging manner.
        - **Unverified Information:** Do not include claims that are not supported by evidence or that come from unreliable sources.
        - **Advertisements:** Do not include any segments that are primarily advertisements for educational products or services.

        **Output Format:**

        Describe the identified clips in natural language, including the start and end times (in seconds) and a brief explanation of why each segment was chosen, focusing on the educational value it provides.

        Use the following JSON structure:

        [
            {{
                "start_time": 678.1,
                "end_time": 723.5,
                "reason": "Clearly explains the concept of compound interest and its importance for long-term savings",
                "quality_score": 92,
                "virality_score": 78,
                "hashtags": "#compoundinterest #investing #education"
            }}
        ]
        """,
    },
}

# --- API Keys and Clients ---
load_dotenv()

GOOGLE_API_KEYS = [
    "AIzaSyA8sIfnnCdpGaGifknf3RBTD-oUN53mwr0",
    "AIzaSyAl6z1texvZtH8X21dxucebouQiNy0JeoM",
    "AIzaSyCRaluyl2U4cfem55hgUuPs54Hb0Sl3qE0",
    "AIzaSyD3OFJZAjXM-xq1BWEj-a3xFNv49e3Lqlw",
    "AIzaSyA5ZqUBzhOPngErO5_tMCmmLANTvX3Ch7o",
    "AIzaSyC9jj3MTJocmHp_7QsgM4UIQ-JWAJWgtfA",
    "AIzaSyDA9NwpiDgLwIUtG_wCBqhsZAvmq1mRlfU"
]
current_google_api_key_index = 0

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Rate Limiting ---
GOOGLE_REQUESTS_PER_MINUTE = 2
GOOGLE_TOKENS_PER_MINUTE = 32000
GOOGLE_REQUESTS_PER_DAY = 50

google_requests_made_this_minute = 0
google_tokens_used_this_minute = 0
google_requests_made_today = 0
last_google_request_time = 0

# --- Directory Setup ---
DOWNLOADS_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)
VIDEOS_DIR.mkdir(exist_ok=True)
GOOGLE_RESPONSES_DIR.mkdir(exist_ok=True)
GROQ_RESPONSES_DIR.mkdir(exist_ok=True)

# --- Helper Functions ---
def initialize_google_api():
    """Initializes the Google API with the current key."""
    global current_google_api_key_index
    if GOOGLE_API_KEYS:
        genai.configure(api_key=GOOGLE_API_KEYS[current_google_api_key_index])

def switch_google_api_key():
    """Switches to the next Google API key."""
    global current_google_api_key_index
    if GOOGLE_API_KEYS:
        current_google_api_key_index = (current_google_api_key_index + 1) % len(GOOGLE_API_KEYS)
        initialize_google_api()
        print(f"Switched to Google API key: {GOOGLE_API_KEYS[current_google_api_key_index]}")

def check_google_rate_limits(prompt):
    """Checks if the prompt would exceed rate limits."""
    global google_requests_made_this_minute, google_tokens_used_this_minute, google_requests_made_today, last_google_request_time
    
    current_time = time.time()
    elapsed_time = current_time - last_google_request_time
    
    if elapsed_time >= 60:
        google_requests_made_this_minute = 0
        google_tokens_used_this_minute = 0
        last_google_request_time = current_time
    
    if google_requests_made_today >= GOOGLE_REQUESTS_PER_DAY:
        print("Reached daily limit for Google API requests.")
        return False 
    
    if google_requests_made_this_minute >= GOOGLE_REQUESTS_PER_MINUTE:
        print("Reached requests per minute limit for Google API.")
        return False
    
    prompt_tokens = len(prompt) # Rough estimate of tokens, adjust if needed
    if google_tokens_used_this_minute + prompt_tokens >= GOOGLE_TOKENS_PER_MINUTE:
        print("Reached tokens per minute limit for Google API.")
        return False

    return True

def update_google_rate_limit_counters(prompt_tokens):
    """Updates the rate limit counters."""
    global google_requests_made_this_minute, google_tokens_used_this_minute, google_requests_made_today
    
    google_requests_made_this_minute += 1
    google_tokens_used_this_minute += prompt_tokens
    google_requests_made_today += 1

def load_settings():
    """Loads settings from the settings file."""
    config = configparser.ConfigParser()
    config.read(SETTINGS_FILE)

    settings = {
        "groq_api_key": config.get("API_KEYS", "groq_api_key", fallback=""),
        "google_api_keys": config.get("API_KEYS", "google_api_keys", fallback=""),
        "use_groq": config.getboolean("SETTINGS", "use_groq", fallback=False),
        "category_prompts": {},
    }

    for category, prompts in CATEGORY_PROMPTS.items():
        settings["category_prompts"][category] = {
            "system": config.get(f"PROMPTS_{category.upper()}", "system", fallback=prompts["system"])
        }

    return settings

def save_settings(settings):
    """Saves settings to the settings file."""
    config = configparser.ConfigParser()
    config["API_KEYS"] = {
        "groq_api_key": settings["groq_api_key"],
        "google_api_keys": settings["google_api_keys"],
    }
    config["SETTINGS"] = {
        "use_groq": str(settings["use_groq"]),
    }

    for category, prompts in settings["category_prompts"].items():
        config[f"PROMPTS_{category.upper()}"] = {
            "system": prompts["system"],
        }

    with open(SETTINGS_FILE, "w") as configfile:
        config.write(configfile)

# --- Transcription Functions (Whisper) ---

print(f"Using device: {DEVICE}")
print(f"Torch dtype: {TORCH_DTYPE}")

# Initialize the Whisper model
print("Loading Whisper model...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
    torch_dtype=TORCH_DTYPE,
    device=DEVICE,
)


def transcribe_audio(audio_path, progress_callback=None):
    """Transcribes the given audio file and saves the transcript."""
    try:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return None

        transcript_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.txt"

        if transcript_path.exists():
            print(f"Transcript already exists: {transcript_path}")
            with open(transcript_path, "r", encoding="utf-8") as f:
                return f.read()

        print(f"\nTranscribing: {audio_path.name}")
        print("This may take a while depending on the audio length...")

        # Verify the audio file is valid
        try:
            import moviepy as mp
            video = mp.VideoFileClip(str(audio_path))
            duration = video.duration
            video.close()
            print(f"Successfully loaded video file. Duration: {duration:.2f} seconds")
        except Exception as ve:
            print(f"Error validating video file: {ve}")
            return None

        start_time = time.time()
        try:
            result = pipe(str(audio_path))
            if not result or "text" not in result:
                print("Error: Transcription returned empty or invalid result")
                return None
        except Exception as te:
            print(f"Error during transcription process: {te}")
            print("Full error details:", traceback.format_exc())
            return None

        # Save the transcript
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("Full Transcription:\n")
                f.write("-" * 80 + "\n")
                f.write(result["text"])
                f.write("\n\n" + "-" * 80 + "\n\n")

                f.write("Timestamped Segments:\n")
                f.write("-" * 80 + "\n")
                total_segments = len(result["chunks"])
                for i, chunk in enumerate(result["chunks"]):
                    timestamp = f"[{chunk['timestamp'][0]:.2f} - {chunk['timestamp'][1]:.2f}]"
                    f.write(f"{timestamp} {chunk['text']}\n")
                    if progress_callback:
                        progress_callback(i + 1, total_segments)

        except Exception as se:
            print(f"Error saving transcript: {se}")
            return None

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\nTranscription completed in {processing_time:.2f} seconds")
        print(f"Saved to: {transcript_path}")

        return result["text"]

    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Full error details:", traceback.format_exc())
        return None

# --- LLM Analysis Functions ---
def refine_with_groq(text):
    """
    Refines the output from the local LLM using Groq's API to produce valid JSON.

    Args:
        text: The output from the local LLM.

    Returns:
        A JSON string with the extracted clip information, or None if an error occurs.
    """
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        response_file = GROQ_RESPONSES_DIR / f"groq_response_{timestamp}.txt"

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates short video clips from transcripts. Your output should be valid JSON."
                },
                {
                    "role": "user",
                    "content": f"Please refine this output into valid JSON that contains an array of clip objects. Each clip should have start_time, end_time, and category fields:\n\n{text}"
                }
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.2,
        )

        # Get the response and save it
        response_text = chat_completion.choices[0].message.content
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response_text)

        print(f"\nGroq response saved to: {response_file}")
        return response_text

    except Exception as e:
        print(f"Error in refine_with_groq: {str(e)}")
        return None

def refine_with_google(text):
    """
    Refines the output from the local LLM using Google's Gemini API to produce valid JSON.

    Args:
        text: The output from the local LLM.

    Returns:
        A JSON string with the extracted clip information, or None if an error occurs.
    """
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        response_file = GOOGLE_RESPONSES_DIR / f"google_response_{timestamp}.txt"
        
        if not check_google_rate_limits(text):
            switch_google_api_key()

        model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
        response = model.generate_content(
            f"Please refine this output into valid JSON that contains an array of clip objects. Each clip should have start_time, end_time, and category fields:\n\n{text}"
        )

        # Update rate limit counters
        update_google_rate_limit_counters(len(text))

        # Save the response
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"\nGoogle response saved to: {response_file}")
        return response.text

    except Exception as e:
        print(f"Error in refine_with_google: {str(e)}")
        if "429" in str(e):
            switch_google_api_key()
        return None
    
def verify_clip_with_groq(clip, full_transcript):
    """
    Uses a Groq LLM to verify if a given clip meets the criteria defined in the system prompt.

    Args:
        clip (dict): A dictionary containing clip information (start_time, end_time, reason).
        full_transcript (str): The full transcript of the audio/video.

    Returns:
        bool: True if the clip is verified as good, False otherwise.
    """
    try:
        segment = extract_segment_from_transcript(
            full_transcript, float(clip["start_time"]), float(clip["end_time"])
        )

        if not segment:
            print("No transcript segment found for clip")
            return False

        prompt = f"""
        You are a social media expert specializing in creating viral content for platforms like TikTok and YouTube Shorts. Your task is to verify if this clip segment would make engaging, viral content.

        **Focus on INSIGHTS:**
        - **Viral Potential:** Look for unique perspectives and thought-provoking ideas be hard on this metric like really think it through
        - **Value:** Check if it offers actionable advice or valuable knowledge
        - **Quality:** Ensure it's engaging and well-structured

        **Clip Details:**
        Duration: {float(clip['end_time']) - float(clip['start_time']):.2f} seconds
        Original Selection Reason: {clip['reason']}

        **Clip Content:**
        {segment}

        **Criteria for Selection:**
        - **High Engagement Potential:** Content that grabs attention quickly
        - **Shareability:** Content people will want to share with others
        - **Informative/Valuable:** Offers unique insights or valuable knowledge
        - **Format Suitability:** Works well for short-form video

        **WHAT NOT TO INCLUDE:**
        - **Non-English Content**
        - **Introductions:** Do not include introductions of the podcast or guest
        - **Outros:** Do not include any outro or closing statements
        - **Advertisements:** No sponsored content or calls to action
        - **Small Talk:** Avoid segments with primarily small talk or filler
        - **Technical Issues:** No audio glitches or difficulties
        - **Inside Jokes:** Avoid content requiring specific context
        - **Offensive Content:** Nothing inappropriate or controversial

        Analyze the clip and provide your assessment in this JSON format:

        [
            {{
                "start_time": {clip['start_time']},
                "end_time": {clip['end_time']},
                "reason": "string",  # Detailed explanation of clip's value
                "quality_score": 85,  # Quality score out of 100
                "virality_score": 75, # Expected virality score out of 100
                "hashtags": "#example #viral #quality"  # Relevant hashtags
            }}
        ]
        """

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a social media expert that verifies video clip quality.",
                },
                {"role": "user", "content": prompt},
            ],
            model=GROQ_VERIFIER_MODEL_NAME,
            temperature=0.1,
            max_tokens=500,
            top_p=1,
            stream=False,
        )

        response_text = chat_completion.choices[0].message.content

        try:
            response = json.loads(response_text)
            if isinstance(response, list) and len(response) > 0:
                clip_data = response[0]
                # Consider the clip approved if both scores are above 70
                return clip_data.get("quality_score", 0) > 70 and clip_data.get("virality_score", 0) > 70
            return False
        except json.JSONDecodeError:
            print(f"Error decoding JSON from Groq verifier response: {response_text}")
            return False

    except Exception as e:
        print(f"Error in verify_clip_with_groq: {str(e)}")
        return False

def verify_clip_with_google(clip, full_transcript):
    """
    Uses a Google Gemini LLM to verify if a given clip meets the criteria defined in the system prompt.

    Args:
        clip (dict): A dictionary containing clip information (start_time, end_time, reason).
        full_transcript (str): The full transcript of the audio/video.

    Returns:
        bool: True if the clip is verified as good, False otherwise.
    """
    try:
        segment = extract_segment_from_transcript(
            full_transcript, float(clip["start_time"]), float(clip["end_time"])
        )

        if not segment:
            print("No transcript segment found for clip")
            return False

        prompt = f"""
        You are a social media expert specializing in creating viral content for platforms like TikTok and YouTube Shorts. Your task is to verify if this clip segment would make engaging, viral content.

        **Focus on INSIGHTS:**
        - **Viral Potential:** Look for unique perspectives and thought-provoking ideas be hard on this metric like really think it through
        - **Value:** Check if it offers actionable advice or valuable knowledge be hard on this metric like really think it through
        - **Quality:** Ensure it's engaging and well-structured be hard on this metric like really think it through

        **Clip Details:**
        Duration: {float(clip['end_time']) - float(clip['start_time']):.2f} seconds
        Original Selection Reason: {clip['reason']}

        **Clip Content:**
        {segment}

        **Criteria for Selection:**
        - **High Engagement Potential:** Content that grabs attention quickly
        - **Shareability:** Content people will want to share with others
        - **Informative/Valuable:** Offers unique insights or valuable knowledge
        - **Format Suitability:** Works well for short-form video

        **WHAT NOT TO INCLUDE:**
        - **Non-English Content**
        - **Introductions:** Do not include introductions of the podcast or guest
        - **Outros:** Do not include any outro or closing statements
        - **Advertisements:** No sponsored content or calls to action
        - **Small Talk:** Avoid segments with primarily small talk or filler
        - **Technical Issues:** No audio glitches or difficulties
        - **Inside Jokes:** Avoid content requiring specific context
        - **Offensive Content:** Nothing inappropriate or controversial

        Analyze the clip and provide your assessment in this JSON format:

        [
            {{
                "start_time": {clip['start_time']},
                "end_time": {clip['end_time']},
                "reason": "string",  # Detailed explanation of clip's value
                "quality_score": 85,  # Quality score out of 100
                "virality_score": 75, # Expected virality score out of 100
                "hashtags": "#example #viral #quality"  # Relevant hashtags
            }}
        ]
        """

        model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
        response = model.generate_content(prompt)
        response_text = response.text

        try:
            response = json.loads(response_text)
            if isinstance(response, list) and len(response) > 0:
                clip_data = response[0]
                # Consider the clip approved if both scores are above 70
                return clip_data.get("quality_score", 0) > 70 and clip_data.get("virality_score", 0) > 70
            return False
        except json.JSONDecodeError:
            print(f"Error decoding JSON from Google verifier response: {response_text}")
            return False

    except Exception as e:
        print(f"Error in verify_clip_with_google: {str(e)}")
        if "429" in str(e):
            switch_google_api_key()
        return False
    
def extract_segment_from_transcript(full_transcript, start_time, end_time):
    """
    Extracts the text segment corresponding to the clip's start and end times from the full transcript.

    Args:
        full_transcript (str): The full transcript text.
        start_time (float): The start time of the clip in seconds.
        end_time (float): The end time of the clip in seconds.

    Returns:
        str: The transcript segment corresponding to the clip.
    """
    lines = full_transcript.split('\n')
    segment_lines = []

    for line in lines:
        match = re.match(r"\[([\d.]+) - ([\d.]+)\] (.+)", line)
        if match:
            line_start, line_end, text = match.groups()
            line_start, line_end = float(line_start), float(line_end)

            if line_start <= end_time and line_end >= start_time:
                segment_lines.append(text)

    return " ".join(segment_lines)

def analyze_transcript_with_llm(transcript_path, category, progress_callback=None):
    """Analyzes the transcript with an LLM to identify potential clips."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        full_transcript_text = f.read()

    transcript_data = full_transcript_text.split('\n')

    # Extract timestamped segments
    timestamped_segments = []
    for line in transcript_data:
        if line.startswith("[") and "]" in line:
            try:
                start_time, end_time = map(
                    float, line[1:].split("]")[0].split(" - ")
                )
                text = line.split("] ", 1)[1].strip()
                timestamped_segments.append(
                    {"start": start_time, "end": end_time, "text": text}
                )
            except ValueError:
                print(f"Skipping invalid line: {line}")

    # Group segments into larger chunks for LLM processing
    chunks = []
    current_chunk = ""
    current_start = None
    for segment in timestamped_segments:
        if current_start is None:
            current_start = segment["start"]

        if len(current_chunk) + len(segment["text"]) < CHUNK_SIZE:
            current_chunk += segment["text"] + " "
        else:
            chunks.append(
                {
                    "start": current_start,
                    "end": segment["end"],
                    "text": current_chunk.strip(),
                }
            )
            current_chunk = segment["text"] + " "
            current_start = segment["start"]

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(
            {
                "start": current_start,
                "end": timestamped_segments[-1]["end"],
                "text": current_chunk.strip(),
            }
        )

    # Get the system prompt based on the selected category
    system_prompt = CATEGORY_PROMPTS[category]["system"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    response_file = GOOGLE_RESPONSES_DIR / f"google_response_{timestamp}.txt"

    identified_clips = []
    total_chunks = len(chunks)

    # Process each chunk with the LLM
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {total_chunks}...")

        # Determine which LLM to use based on the flag
        settings = load_settings()
        use_groq = settings["use_groq"]

        if use_groq:
            # --- Groq processing ---
            # Generate a unique temporary model name for each chunk
            temp_model_name = f"{LOCAL_MODEL_NAME}-{uuid.uuid4()}"

            # Duplicate the base model to the temporary model
            try:
                ollama.create(
                    model=temp_model_name,
                    modelfile=f"""
                    FROM {LOCAL_MODEL_NAME}
                    """,
                )
                print(f"Created temporary model: {temp_model_name}")
            except Exception as e:
                print(f"Error creating temporary model: {e}")
                continue

            user_prompt = f"""
            Analyze the following transcript segment:

            Text: \"{chunk["text"]}\"
            Start Time: {chunk["start"]}
            End Time: {chunk["end"]}

            Identify any parts that meet the criteria outlined in the system prompt, ensuring each identified clip is at least {MIN_CLIP_LENGTH} seconds and at most {MAX_CLIP_LENGTH} seconds long. Only include clips with exceptionally high potential for virality.

            Format your response as a JSON array with the following structure:

            [
                {{
                    "start_time": [start time in seconds],
                    "end_time": [end time in seconds],
                    "reason": "[reason for selection]",
                    "quality_score": [quality score out of 100],
                    "virality_score": [virality score out of 100],
                    "hashtags": "[relevant hashtags]"
                }}
            ]
            """

            response = ollama.chat(
                model=temp_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            llm_response_content = response["message"]["content"]
            print(f"Raw LLM response for chunk {i+1}: {llm_response_content}")

            # --- Removed refining ---
            refined_json = llm_response_content 

            # Delete the temporary model after processing the chunk
            try:
                ollama.delete(model=temp_model_name)
                print(f"Deleted temporary model: {temp_model_name}")
            except Exception as e:
                print(f"Error deleting temporary model: {e}")

        else:
            # --- Google Gemini processing ---
            if not check_google_rate_limits(chunk["text"]):
                switch_google_api_key()
                
            user_prompt = f"""
            Analyze the following transcript segment:

            Text: \"{chunk["text"]}\"
            Start Time: {chunk["start"]}
            End Time: {chunk["end"]}

            Identify any parts that meet the criteria outlined in the system prompt, ensuring each identified clip is at least {MIN_CLIP_LENGTH} seconds and at most {MAX_CLIP_LENGTH} seconds long. Only include clips with exceptionally high potential for virality.
            
            Format your response as a JSON array with the following structure:

            [
                {{
                    "start_time": [start time in seconds],
                    "end_time": [end time in seconds],
                    "reason": "[reason for selection]",
                    "quality_score": [quality score out of 100],
                    "virality_score": [virality score out of 100],
                    "hashtags": "[relevant hashtags]"
                }}
            ]
            """

            model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            # Update rate limit counters
            update_google_rate_limit_counters(len(user_prompt))

            llm_response_content = response.text
            print(f"Raw LLM response for chunk {i+1}: {llm_response_content}")

            # --- Removed refining ---
            refined_json = llm_response_content

        if refined_json:
            print(f"Refined JSON for chunk {i+1}: {refined_json}")
            try:
                # Remove leading/trailing whitespace and any extra spaces within the JSON string
                cleaned_json = re.sub(r"^\s+|\s+$", "", refined_json, flags=re.MULTILINE)  # Remove leading/trailing whitespace
                cleaned_json = re.sub(r"  +", " ", cleaned_json)  # Remove extra spaces

                clips_data = json.loads(cleaned_json)

                if isinstance(clips_data, list):  # Ensure clips_data is a list
                    for clip in clips_data:
                        # Validate and adjust clip times
                        clip["start_time"] = max(0.0, float(clip.get("start_time", 0.0)))
                        clip["end_time"] = max(0.0, float(clip.get("end_time", 0.0)))
                        clip_duration = clip["end_time"] - clip["start_time"]

                        if clip_duration < MIN_CLIP_LENGTH:
                            clip["end_time"] = clip["start_time"] + MIN_CLIP_LENGTH
                        elif clip_duration > MAX_CLIP_LENGTH:
                            clip["end_time"] = clip["start_time"] + MAX_CLIP_LENGTH

                        # Adjust timestamps to be relative to the full transcript
                        clip["start_time"] += chunk["start"]
                        clip["end_time"] += chunk["start"]

                        # Ensure the clip length is within the desired range
                        if (MIN_CLIP_LENGTH <= clip["end_time"] - clip["start_time"] <= MAX_CLIP_LENGTH):
                            # Remove the verifier for now since you want the direct output
                            identified_clips.append(clip)
                            

                        if len(identified_clips) >= MAX_CLIPS_TO_FIND:
                            break
                else:
                    print(f"Refined JSON for chunk {i+1} is not a list as expected.")

                if len(identified_clips) >= MAX_CLIPS_TO_FIND:
                    break

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from response for chunk {i+1}: {e}")
                print(f"Problematic JSON string: {refined_json}")
        else:
            print(f"Could not refine output for chunk {i+1}.")

        if progress_callback:
            progress_callback(i + 1, total_chunks)

    # Save identified clips
    clips_file_path = CLIPS_DIR / f"{transcript_path.stem}_clips.json"
    with open(clips_file_path, "w", encoding="utf-8") as f:
        json.dump(identified_clips, f, indent=4)

    return identified_clips
# --- Video Processing Functions ---




def download_youtube_video(url):
    """Downloads a YouTube video and returns the paths to both the video and audio files."""
    try:
        # Configure yt-dlp options for video
        video_opts = {
            'format': 'best[ext=mp4]',  # Download best quality MP4
            'outtmpl': str(VIDEOS_DIR / '%(title)s.%(ext)s'),  # Save to videos directory
            'quiet': True,
            'no_warnings': True,
        }
        
        # Configure yt-dlp options for audio
        audio_opts = {
            'format': 'bestaudio/best',  # Download best quality audio
            'outtmpl': str(AUDIO_DIR / '%(title)s.%(ext)s'),  # Save to audio directory
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        # Download the video
        with YoutubeDL(video_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = VIDEOS_DIR / f"{info['title']}.mp4"
        
        # Download the audio
        with YoutubeDL(audio_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = AUDIO_DIR / f"{info['title']}.mp3"
            
        return str(video_path), str(audio_path)
            
    except Exception as e:
        raise Exception(f"Error downloading YouTube video: {str(e)}")

# --- UI Class ---

class App(ctk.CTk):
    """Main application class using customtkinter."""

    def __init__(self):
        super().__init__()

        self.title("AI Video Clipper")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")

        # Load settings
        self.settings = load_settings()

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        # Logo and Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="AI Video Clipper",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Source Selection
        self.source_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Video Source:",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.source_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.source_var = ctk.StringVar(value="local")
        self.local_radio = ctk.CTkRadioButton(
            self.sidebar_frame,
            text="Local Video",
            variable=self.source_var,
            value="local",
            command=self.on_source_change
        )
        self.local_radio.grid(row=2, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.youtube_radio = ctk.CTkRadioButton(
            self.sidebar_frame,
            text="YouTube URL",
            variable=self.source_var,
            value="youtube",
            command=self.on_source_change
        )
        self.youtube_radio.grid(row=3, column=0, padx=20, pady=(5, 20), sticky="w")

        # URL Entry (initially hidden)
        self.url_entry = ctk.CTkEntry(
            self.sidebar_frame,
            placeholder_text="Enter YouTube URL...",
            width=160
        )
        self.url_entry.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="w")
        self.url_entry.grid_remove()  # Hidden by default

        # Category Selection
        self.category_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Category:",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.category_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.category_optionmenu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=list(CATEGORY_PROMPTS.keys()),
            command=self.on_category_change,
            width=160
        )
        self.category_optionmenu.grid(row=6, column=0, padx=20, pady=(5, 20), sticky="w")

        # Groq Toggle
        self.use_groq_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Use Groq:",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.use_groq_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.use_groq_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="",
            command=self.on_use_groq_change,
        )
        self.use_groq_switch.grid(row=8, column=0, padx=20, pady=(5, 20), sticky="w")
        self.use_groq_switch.select() if self.settings["use_groq"] else self.use_groq_switch.deselect()

        # API Keys Button
        self.api_keys_button = ctk.CTkButton(
            self.sidebar_frame,
            text="API Keys",
            command=self.open_api_keys_window,
            width=160
        )
        self.api_keys_button.grid(row=9, column=0, padx=20, pady=(10, 10))

        # Open Downloads Folder Button
        self.open_downloads_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Open Downloads",
            command=self.open_downloads_folder,
            width=160
        )
        self.open_downloads_button.grid(row=10, column=0, padx=20, pady=(0, 20))

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(2, weight=2)
        self.main_frame.grid_rowconfigure(3, weight=2)

        # Files List Frame
        self.files_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.files_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.files_frame.grid_columnconfigure(0, weight=1)

        self.files_label = ctk.CTkLabel(
            self.files_frame,
            text="Available Files",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.files_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        # Audio Files Listbox with modern styling
        self.audio_listbox = tk.Listbox(
            self.files_frame,
            font=("Segoe UI", 12),
            bg="#2b2b2b",
            fg="#ffffff",
            selectbackground="#3a3a3a",
            selectforeground="#ffffff",
            borderwidth=0,
            highlightthickness=0
        )
        self.audio_listbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.populate_audio_list()

        # Buttons Frame with modern styling
        self.buttons_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.buttons_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        button_width = 120
        self.transcribe_button = ctk.CTkButton(
            self.buttons_frame,
            text="Transcribe",
            command=self.transcribe_selected_audio,
            width=button_width,
            fg_color="#2ecc71",
            hover_color="#27ae60"
        )
        self.transcribe_button.pack(side="left", padx=10, pady=10)

        self.process_button = ctk.CTkButton(
            self.buttons_frame,
            text="Process",
            command=self.process_selected_audio,
            width=button_width,
            fg_color="#3498db",
            hover_color="#2980b9"
        )
        self.process_button.pack(side="left", padx=10, pady=10)

        self.reset_button = ctk.CTkButton(
            self.buttons_frame,
            text="Reset",
            command=self.reset_ui,
            width=button_width
        )
        self.reset_button.pack(side="left", padx=10, pady=10)

        self.clear_logs_button = ctk.CTkButton(
            self.buttons_frame,
            text="Clear Logs",
            command=self.clear_logs,
            width=button_width
        )
        self.clear_logs_button.pack(side="left", padx=10, pady=10)

        # Prompt Section
        self.prompt_label = ctk.CTkLabel(
            self.main_frame,
            text=f"Prompt ({self.category_optionmenu.get()}):",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.prompt_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nw")

        self.prompt_textbox = ctk.CTkTextbox(
            self.main_frame,
            wrap="word",
            font=("Segoe UI", 12),
            corner_radius=10
        )
        self.prompt_textbox.grid(row=2, column=0, padx=10, pady=(30, 0), sticky="nsew")
        self.update_prompt_textbox()

        # Reset Prompt Button
        self.reset_prompt_button = ctk.CTkButton(
            self.main_frame,
            text="Reset to Default",
            command=self.reset_prompt,
            fg_color="#e67e22",
            hover_color="#d35400",
            width=120
        )
        self.reset_prompt_button.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="se")
        
        # --- Log Textbox ---
        self.log_label = ctk.CTkLabel(
            self.main_frame,
            text="Logs:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.log_label.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="nw")

        self.log_textbox = ctk.CTkTextbox(
            self.main_frame,
            wrap="word",
            font=("Segoe UI", 12),
            corner_radius=10,
            state="disabled"
        )
        self.log_textbox.grid(row=3, column=0, padx=10, pady=(30, 10), sticky="nsew")

        # Progress Bar with modern styling
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar",
                       troughcolor='#2b2b2b',
                       background='#3498db',
                       thickness=10)
        
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            orient="horizontal",
            mode="determinate",
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Bind double-click to audio listbox
        self.audio_listbox.bind("<Double-Button-1>", self.transcribe_selected_audio)
        
    def on_source_change(self):
        if self.source_var.get() == "youtube":
            self.url_entry.grid()
            self.audio_listbox.config(state="disabled")
        else:
            self.url_entry.grid_remove()
            self.audio_listbox.config(state="normal")
            
    def on_category_change(self, *args):
        """Handles category change event."""
        self.update_prompt_textbox()

    def on_use_groq_change(self):
        """Handles Groq toggle change event."""
        self.settings["use_groq"] = self.use_groq_switch.get()
        save_settings(self.settings)
        self.log_message(f"Groq {'enabled' if self.settings['use_groq'] else 'disabled'}")

    def open_api_keys_window(self):
        """Opens a window to manage API keys."""
        api_keys_window = ctk.CTkToplevel(self)
        api_keys_window.title("API Keys")
        api_keys_window.geometry("400x300")

        # Groq API Key
        groq_api_key_label = ctk.CTkLabel(api_keys_window, text="Groq API Key:")
        groq_api_key_label.pack(pady=(10, 0))
        groq_api_key_entry = ctk.CTkEntry(api_keys_window)
        groq_api_key_entry.insert(0, self.settings["groq_api_key"])
        groq_api_key_entry.pack()

        # Google API Keys
        google_api_keys_label = ctk.CTkLabel(api_keys_window, text="Google API Keys (comma-separated):")
        google_api_keys_label.pack(pady=(10, 0))
        google_api_keys_entry = ctk.CTkEntry(api_keys_window)
        google_api_keys_entry.insert(0, self.settings["google_api_keys"])
        google_api_keys_entry.pack()

        def save_api_keys():
            self.settings["groq_api_key"] = groq_api_key_entry.get()
            self.settings["google_api_keys"] = google_api_keys_entry.get()
            save_settings(self.settings)
            self.log_message("API keys saved.")
            api_keys_window.destroy()

        save_button = ctk.CTkButton(api_keys_window, text="Save", command=save_api_keys)
        save_button.pack(pady=20)

    def open_downloads_folder(self):
        """Opens the downloads folder in the file explorer."""
        try:
            os.startfile(DOWNLOADS_DIR)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open downloads folder: {e}")

    def populate_audio_list(self):
        """Populates the audio listbox with available MP3 files."""
        self.audio_listbox.delete(0, tk.END)
        audio_files = list(AUDIO_DIR.glob("*.mp3"))
        for file in audio_files:
            self.audio_listbox.insert(tk.END, file.name)

    def transcribe_selected_audio(self, event=None):
        """Transcribes the selected audio file."""
        if self.source_var.get() == "youtube":
            url = self.url_entry.get().strip()
            if not url:
                messagebox.showerror("Error", "Please enter a YouTube URL")
                return
            try:
                self.log_message("Downloading YouTube video...")
                video_path, audio_path = download_youtube_video(url)
                self.log_message(f"Downloaded video to: {video_path}")
                self.log_message(f"Downloaded audio to: {audio_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
        else:
            selection = self.audio_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select an audio file")
                return
            video_path = os.path.join(AUDIO_DIR, self.audio_listbox.get(selection[0]))

        def transcription_task():
            try:
                self.transcribe_button.configure(state="disabled")
                self.process_button.configure(state="disabled")
                
                self.log_message(f"Transcribing: {os.path.basename(video_path)}")
                transcript_path = transcribe_audio(video_path, self.update_progress)
                
                self.log_message(f"Transcription completed: {transcript_path}")
                plyer.notification.notify(
                    title='Transcription Complete',
                    message=f'The file has been transcribed successfully!',
                    app_icon=None,
                    timeout=10,
                )
                
                self.transcribe_button.configure(state="normal")
                self.process_button.configure(state="normal")
                
            except Exception as e:
                self.log_message(f"Error during transcription: {str(e)}")
                messagebox.showerror("Error", f"Transcription failed: {str(e)}")
                self.transcribe_button.configure(state="normal")
                self.process_button.configure(state="normal")

        threading.Thread(target=transcription_task, daemon=True).start()
        
    def process_selected_audio(self):
        """Processes the selected audio file (transcription and analysis)."""
        selected_indices = self.audio_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select an audio file")
            return
        
        selected_files = [self.audio_listbox.get(i) for i in selected_indices]
        category = self.category_optionmenu.get()

        for file_name in selected_files:
            audio_path = AUDIO_DIR / file_name
            transcript_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.txt"
            video_path = VIDEOS_DIR / f"{audio_path.stem}.mp4"

            if not transcript_path.exists():
                self.log_message(f"Transcript not found for {file_name}. Please transcribe first.")
                continue

            self.log_message(f"Starting processing for: {file_name}")
            self.transcribe_button.configure(state="disabled")
            self.process_button.configure(state="disabled")

            def analysis_task():
                try:
                    analyze_transcript_with_llm(
                        transcript_path,
                        category,
                        progress_callback=lambda current, total: self.update_progress(
                            current, total, f"Analyzing {file_name}"
                        ),
                    )
                    self.log_message(f"Analysis complete for: {file_name}")
                    plyer.notification.notify(
                        title="Analysis Complete",
                        message=f"Analysis for {file_name} is complete.",
                        timeout=5,
                    )
                except Exception as e:
                    self.log_message(f"Error during analysis of {file_name}: {e}")

                try:
                    # Create and organize clips
                    clips_json_file = CLIPS_DIR / f"{audio_path.stem}_clips.json"
                    if clips_json_file.exists():
                        self.log_message(f"Creating clips for {file_name}")
                        create_organized_clips(clips_json_file, video_path, CLIPS_DIR, transcript_path)
                        self.log_message(f"Clip creation complete for {file_name}")
                        plyer.notification.notify(
                            title="Clip Creation Complete",
                            message=f"Clips for {file_name} created successfully.",
                            timeout=5,
                        )
                    else:
                        self.log_message(f"Could not find clips JSON for {file_name}. Clip creation skipped.")
                except Exception as e:
                    self.log_message(f"Error during clip creation for {file_name}: {e}")
                finally:
                    self.after(0, self.enable_buttons)  # Re-enable buttons

            threading.Thread(target=analysis_task).start()

    def enable_buttons(self):
        """Enables the transcribe and process buttons."""
        self.transcribe_button.configure(state="normal")
        self.process_button.configure(state="normal")

    def reset_ui(self):
        """Resets the UI to its initial state."""
        self.audio_listbox.selection_clear(0, tk.END)
        self.progress_bar["value"] = 0
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", tk.END)
        self.log_textbox.configure(state="disabled")
        self.populate_audio_list()  # Refresh the audio list
        self.log_message("UI reset.")

    def clear_logs(self):
        """Clears the log textbox."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", tk.END)
        self.log_textbox.configure(state="disabled")
        self.log_message("Logs cleared.")

    def update_progress(self, current, total, message=""):
        """Updates the progress bar and log message."""
        progress_percent = (current / total) * 100
        self.progress_bar["value"] = progress_percent
        self.log_message(f"{message} - Progress: {current}/{total} ({progress_percent:.1f}%)")
        self.update_idletasks()  # Update the UI

    def log_message(self, message):
        """Logs a message to the log textbox."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_textbox.configure(state="normal")  # Enable editing
        self.log_textbox.insert(tk.END, formatted_message)
        self.log_textbox.configure(state="disabled")  # Disable editing
        self.log_textbox.see(tk.END)  # Scroll to the end

    def update_prompt_textbox(self):
        """Updates the prompt textbox with the current category's prompt."""
        category = self.category_optionmenu.get()
        system_prompt = self.settings["category_prompts"][category]["system"]
        self.prompt_textbox.delete("1.0", tk.END)
        self.prompt_textbox.insert("1.0", system_prompt)
        self.prompt_label.configure(text=f"Prompt ({category}):")

    def reset_prompt(self):
        """Resets the current category's prompt to its default value."""
        category = self.category_optionmenu.get()
        default_prompt = CATEGORY_PROMPTS[category]["system"]
        self.settings["category_prompts"][category]["system"] = default_prompt
        save_settings(self.settings)
        self.update_prompt_textbox()
        self.log_message(f"Prompt for {category} reset to default.")

def main():
    initialize_google_api()
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
