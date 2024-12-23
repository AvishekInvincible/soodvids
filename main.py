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

# --- Configuration ---
DOWNLOADS_DIR = Path("downloads")
AUDIO_DIR = DOWNLOADS_DIR / "audio"
TRANSCRIPTS_DIR = DOWNLOADS_DIR / "transcripts"
CLIPS_DIR = DOWNLOADS_DIR / "clips"
VIDEOS_DIR = DOWNLOADS_DIR / "videos"
GOOGLE_RESPONSES_DIR = DOWNLOADS_DIR / "google_responses" # Directory for Google responses
GROQ_RESPONSES_DIR = DOWNLOADS_DIR / "groq_responses" # Directory for Groq responses

# LLM Settings
LOCAL_MODEL_NAME = "llama3:70b-instruct-q2_K"
GROQ_MODEL_NAME = "llama-3.3-70b-specdec"

GROQ_VERIFIER_MODEL_NAME = GROQ_MODEL_NAME 
GOOGLE_MODEL_NAME = "gemini-1.5-pro"
CHUNK_SIZE = 12000
MIN_CLIP_LENGTH = 20
MAX_CLIP_LENGTH = 59
MAX_CLIPS_TO_FIND = 10

# Whisper Model Settings
MODEL_ID = "openai/whisper-large-v3-turbo"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# API Keys and Clients
load_dotenv()

current_google_api_key_index = 0  # Start with the first key

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Flags ---
USE_GROQ = False  # Set to True to use Groq, False to use Google Gemini

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
    genai.configure(api_key=GOOGLE_API_KEYS[current_google_api_key_index])

def switch_google_api_key():
    """Switches to the next Google API key."""
    global current_google_api_key_index
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


def transcribe_audio(audio_path):
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

        start_time = time.time()
        result = pipe(str(audio_path))

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("Full Transcription:\n")
            f.write("-" * 80 + "\n")
            f.write(result["text"])
            f.write("\n\n" + "-" * 80 + "\n\n")

            f.write("Timestamped Segments:\n")
            f.write("-" * 80 + "\n")
            for chunk in result["chunks"]:
                timestamp = f"[{chunk['timestamp'][0]:.2f} - {chunk['timestamp'][1]:.2f}]"
                f.write(f"{timestamp} {chunk['text']}\n")

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\nTranscription completed in {processing_time:.2f} seconds")
        print(f"Saved to: {transcript_path}")

        return result["text"]

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def list_audio_files():
    print("\nAvailable audio files:")
    audio_files = list(AUDIO_DIR.glob("*.mp3"))
    if not audio_files:
        print("No audio files found in downloads/audio directory")
        return []

    for i, file in enumerate(audio_files, 1):
        file_size = file.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"{i}. {file.name} ({file_size:.1f} MB)")
    return audio_files


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
        start_time = clip['start_time']
        end_time = clip['end_time']

        # Extract the relevant segment from the full transcript
        clip_text = extract_segment_from_transcript(full_transcript, start_time, end_time)

        verification_prompt = f"""
        You are a social media expert tasked with verifying the quality of short video clips extracted from a podcast transcript. 
        Your goal is to determine if the provided clip meets the criteria for engaging, insightful, and viral content suitable for platforms like TikTok and YouTube Shorts.

        **Criteria for a High-Quality Clip:**

        - **Insightful:** Offers unique perspectives, thought-provoking ideas, valuable knowledge, or actionable advice.
        - **Engaging:** Likely to grab attention quickly, keep viewers hooked, and spark curiosity.
        - **Shareable:** The kind of content people would want to share with friends or followers because it's insightful, surprising, or thought-provoking.
        - **Informative/Valuable:** Provides unique insights, practical advice, or valuable knowledge that the average viewer would find beneficial.
        - **Suitable for Short-Form Video:**  Fits the format of TikTok/YouTube Shorts (at least {MIN_CLIP_LENGTH} seconds long and a maximum of {MAX_CLIP_LENGTH} seconds long). The clip should end naturally and not cut off abruptly.

        **WHAT NOT TO INCLUDE:**

        - **Non-English:** Any non-English content.
        - **Introductions:** Introductions of the podcast or the guest.
        - **Outros:** Any outro or closing statements.
        - **Advertisements or Sponsorships:** Segments that are clearly advertisements, sponsored content, or calls to action related to sponsors.
        - **Small Talk or Filler:** Segments that are primarily small talk, off-topic digressions, or contain excessive filler words/phrases (e.g., "um," "like," "you know").
        - **Repetitive Content:** Clips that are overly repetitive or reiterate points already made in other selected clips.
        - **Technical Difficulties:** Parts where there are technical issues, audio glitches, or extended periods of silence.
        - **Inside Jokes or Overly Niche References:** Content that relies heavily on inside jokes, extremely niche references, or specialized terminology that the average viewer would not understand.
        - **Personal Anecdotes (Unless Highly Relevant and Insightful):** Personal stories or anecdotes that are not exceptionally engaging, do not directly illustrate a key insight, or are not relatable to a broad audience.
        - **Unclear or Mumbled Speech**: Sections where the speech is difficult to understand due to mumbling, fast talking, or heavy accents.
        - **Offensive or Controversial Content**: Anything that could be considered offensive, insensitive, or overly controversial for a general audience.

        **Clip Information:**

        - Start Time: {start_time} seconds
        - End Time: {end_time} seconds
        - Transcript Segment: "{clip_text}"

        **Task:**

        Based on the criteria above, does this clip qualify as a high-quality, insightful, and engaging short video clip? 

        Provide a brief explanation of your reasoning, highlighting strengths or weaknesses of the clip in relation to the criteria.

        **Output Format:**

        {{
            "approved": true/false,  // true if the clip meets the criteria, false otherwise
            "reason": "string"    // Your reasoning for approving or rejecting the clip
        }}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates and approves short video clips based on specific criteria."
                },
                {
                    "role": "user",
                    "content": verification_prompt
                }
            ],
            model=GROQ_VERIFIER_MODEL_NAME,
            temperature=0.2,  # Lower temperature for more focused evaluation
        )

        response_text = chat_completion.choices[0].message.content
        print(f"Verification response: {response_text}")

        # Save the response for debugging/logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        response_file = GROQ_RESPONSES_DIR / f"groq_verifier_response_{timestamp}.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response_text)

        # Extract the 'approved' status
        try:
            response_json = json.loads(response_text)
            return response_json.get("approved", False)  # Default to False if not found
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
        start_time = clip['start_time']
        end_time = clip['end_time']

        # Extract the relevant segment from the full transcript
        clip_text = extract_segment_from_transcript(full_transcript, start_time, end_time)
        
        if not check_google_rate_limits(clip_text):
            switch_google_api_key()

        verification_prompt = f"""
        You are a social media expert tasked with verifying the quality of short video clips extracted from a podcast transcript. 
        Your goal is to determine if the provided clip meets the criteria for engaging, insightful, and viral content suitable for platforms like TikTok and YouTube Shorts.

        **Criteria for a High-Quality Clip:**

        - **Insightful:** Offers unique perspectives, thought-provoking ideas, valuable knowledge, or actionable advice.
        - **Engaging:** Likely to grab attention quickly, keep viewers hooked, and spark curiosity.
        - **Shareable:** The kind of content people would want to share with friends or followers because it's insightful, surprising, or thought-provoking.
        - **Informative/Valuable:** Provides unique insights, practical advice, or valuable knowledge that the average viewer would find beneficial.
        - **Suitable for Short-Form Video:**  Fits the format of TikTok/YouTube Shorts (at least {MIN_CLIP_LENGTH} seconds long and a maximum of {MAX_CLIP_LENGTH} seconds long). The clip should end naturally and not cut off abruptly.

        **WHAT NOT TO INCLUDE:**

        - **Non-English:** Any non-English content.
        - **Introductions:** Introductions of the podcast or the guest.
        - **Outros:** Any outro or closing statements.
        - **Advertisements or Sponsorships:** Segments that are clearly advertisements, sponsored content, or calls to action related to sponsors.
        - **Small Talk or Filler:** Segments that are primarily small talk, off-topic digressions, or contain excessive filler words/phrases (e.g., "um," "like," "you know").
        - **Repetitive Content:** Clips that are overly repetitive or reiterate points already made in other selected clips.
        - **Technical Difficulties:** Parts where there are technical issues, audio glitches, or extended periods of silence.
        - **Inside Jokes or Overly Niche References:** Content that relies heavily on inside jokes, extremely niche references, or specialized terminology that the average viewer would not understand.
        - **Personal Anecdotes (Unless Highly Relevant and Insightful):** Personal stories or anecdotes that are not exceptionally engaging, do not directly illustrate a key insight, or are not relatable to a broad audience.
        - **Unclear or Mumbled Speech**: Sections where the speech is difficult to understand due to mumbling, fast talking, or heavy accents.
        - **Offensive or Controversial Content**: Anything that could be considered offensive, insensitive, or overly controversial for a general audience.

        **Clip Information:**

        - Start Time: {start_time} seconds
        - End Time: {end_time} seconds
        - Transcript Segment: "{clip_text}"

        **Task:**

        Based on the criteria above, does this clip qualify as a high-quality, insightful, and engaging short video clip? 

        Provide a brief explanation of your reasoning, highlighting strengths or weaknesses of the clip in relation to the criteria.

        **Output Format:**

        {{
            "approved": true/false,  // true if the clip meets the criteria, false otherwise
            "reason": "string"    // Your reasoning for approving or rejecting the clip
        }}
        """
        model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
        response = model.generate_content(verification_prompt)

        # Update rate limit counters
        update_google_rate_limit_counters(len(verification_prompt))

        response_text = response.text
        print(f"Verification response: {response_text}")

        # Save the response for debugging/logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        response_file = GOOGLE_RESPONSES_DIR / f"google_verifier_response_{timestamp}.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response_text)

        # Extract the 'approved' status
        try:
            response_json = json.loads(response_text)
            return response_json.get("approved", False)  # Default to False if not found
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

def analyze_transcript_with_llm(transcript_path):
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

    # --- Enhanced LLM Prompt ---
    system_prompt = f"""
    You are a social media expert specializing in creating viral content for platforms like TikTok and YouTube Shorts. Your task is to analyze the provided podcast transcript segments and extract {MAX_CLIPS_TO_FIND} timestamps of **insightful** parts that are highly likely to be engaging, informative, or go viral as short-form videos.

    **Focus on INSIGHTS:**

    - **Prioritize clips that offer unique perspectives, thought-provoking ideas, valuable knowledge, or actionable advice.** These are the types of clips that tend to perform well and generate discussion on social media.
    - **Think "Aha!" moments:** Look for segments where the speaker reveals something new, challenges conventional wisdom, or provides a fresh take on a familiar topic.
    - **Practical takeaways:** Select clips that offer viewers something they can apply to their own lives or learn from.

    **Criteria for Selection:**

    - **High Engagement Potential:**  Prioritize content that is likely to grab attention quickly, keep viewers hooked, and spark curiosity. **The start should be a hook**
    - **Shareability:** Select segments that people will want to share with their friends or followers because they're insightful, surprising, or thought-provoking.
    - **Informative/Valuable:** Choose segments that offer unique insights, practical advice, or valuable knowledge that the average viewer would find beneficial.
    - **TikTok/YouTube Shorts Format:** The identified clips should be suitable for short-form video platforms. Each clip must be at least {MIN_CLIP_LENGTH} seconds long and a maximum of {MAX_CLIP_LENGTH} seconds long. **Ensure the clip ends naturally and doesn't cut off abruptly.**

    **WHAT NOT TO INCLUDE:**

    - **Non-English:** Do not include any non-English content.
    - **Introductions:** Do not include introductions of the podcast or the guest. Focus on the core content.
    - **Outros:** Do not include any outro or closing statements.
    - **Advertisements or Sponsorships:** Exclude any segments that are clearly advertisements, sponsored content, or calls to action related to sponsors.
    - **Small Talk or Filler:** Avoid segments that are primarily small talk, off-topic digressions, or contain excessive filler words/phrases
    - **Repetitive Content:** Do not select clips that are overly repetitive or reiterate points already made in other selected clips.
    - **Technical Difficulties:** Exclude any parts where there are technical issues, audio glitches, or extended periods of silence.
    - **Inside Jokes or Overly Niche References:** Avoid content that relies heavily on inside jokes, extremely niche references, or specialized terminology that the average viewer would not understand.
    - **Personal Anecdotes (Unless Highly Relevant and Insightful):** Only include personal stories or anecdotes if they are exceptionally engaging, directly illustrate a key insight, and are relatable to a broad audience.
    - **Unclear or Mumbled Speech**: Do not include sections where the speech is difficult to understand due to mumbling, fast talking, or heavy accents.
    - **Offensive or Controversial Content**: Avoid selecting anything that could be considered offensive, insensitive, or overly controversial for a general audience.

    **Output Format:**

    Describe the identified clips in natural language, including the start and end times (in seconds) and a brief, specific explanation of why each segment was chosen, focusing on the **insight** it provides.

    Use a numbered list format like this (Each clip must be at least {MIN_CLIP_LENGTH} seconds long and a maximum of {MAX_CLIP_LENGTH} seconds long. Ideally, aim for clips around 30-60 seconds, but shorter, impactful clips are also acceptable.):

    1.  Clip 1: [Clip Title/Description] (Start Time: [start_time], End Time: [end_time])
        [Reason for choosing this clip, highlighting the insight offered]
    2.  Clip 2: [Clip Title/Description] (Start Time: [start_time], End Time: [end_time])
        [Reason for choosing this clip, highlighting the insight offered]

    Format your response as JSON with the following structure:


    [
        {{
            "start_time": 54.16,  // Start time in seconds
            "end_time": 80.1,    // End time in seconds
            "reason": "string"  // Why this clip is insightful and would be engaging
        }}
    ] """

    identified_clips = []
    total_chunks = len(chunks)

    # Process each chunk with the LLM
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {total_chunks}...")

        # Determine which LLM to use based on the flag
        if USE_GROQ:
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
            Analyze the following podcast transcript segment:

            Text: \"{chunk["text"]}\"
            Start Time: {chunk["start"]}
            End Time: {chunk["end"]}

            Identify any parts that meet the criteria outlined in the system prompt, ensuring each identified clip is at least {MIN_CLIP_LENGTH} seconds and at most {MAX_CLIP_LENGTH} seconds long.
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

            # Refine the output with Groq
            refined_json = refine_with_groq(llm_response_content)
            
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
            Analyze the following podcast transcript segment:

            Text: \"{chunk["text"]}\"
            Start Time: {chunk["start"]}
            End Time: {chunk["end"]}

            Identify any parts that meet the criteria outlined in the system prompt, ensuring each identified clip is at least {MIN_CLIP_LENGTH} seconds and at most {MAX_CLIP_LENGTH} seconds long.
            """

            model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            
            # Update rate limit counters
            update_google_rate_limit_counters(len(user_prompt))

            llm_response_content = response.text
            print(f"Raw LLM response for chunk {i+1}: {llm_response_content}")

            # Refine the output with Google Gemini
            refined_json = refine_with_google(llm_response_content)

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
                            # Verify with the chosen verifier
                            if USE_GROQ:
                                if verify_clip_with_groq(clip, full_transcript_text):
                                    identified_clips.append(clip)
                                else:
                                    print(f"Clip rejected by Groq verifier: {clip}")
                            else:
                                if verify_clip_with_google(clip, full_transcript_text):
                                    identified_clips.append(clip)
                                else:
                                    print(f"Clip rejected by Google verifier: {clip}")

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

    # Save identified clips
    clips_file_path = CLIPS_DIR / f"{transcript_path.stem}_clips.json"
    with open(clips_file_path, "w", encoding="utf-8") as f:
        json.dump(identified_clips, f, indent=4)

    return identified_clips

# --- Video Processing Functions ---

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS,mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def get_transcript_segments(transcript_path, start_time, end_time):
    """Extract transcript segments for the given time range."""
    segments = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find all timestamp segments
    pattern = r'\[([\d.]+) - ([\d.]+)\] (.+?)\n'
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        seg_start = float(match.group(1))
        seg_end = float(match.group(2))
        text = match.group(3).strip()

        # Check if this segment overlaps with our clip
        if seg_start <= end_time and seg_end >= start_time:
            # Adjust text timing to be relative to clip start
            adj_start = max(0, seg_start - start_time)
            adj_end = min(end_time - start_time, seg_end - start_time)
            segments.append({
                'start': adj_start,
                'end': adj_end,
                'text': text
            })

    return segments

def create_subtitle_file(segments, output_path):
    """
    Create an SRT subtitle file with:
    - Word-level timestamps
    - Emphasis (yellow color) on the last word of each subtitle
    - Max 3 words per line, except for the last line which can have up to 5
    - Centered and positioned towards the bottom
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        subtitle_index = 1
        for segment in segments:
            words = segment['text'].split()
            segment_start_time = segment['start']
            segment_duration = segment['end'] - segment['start']
            time_per_word = segment_duration / len(words) if len(words) > 0 else 0

            # Group words into chunks of up to 3 words (5 for the last line)
            for i in range(0, len(words), 3):
                chunk = words[i:i+3]
                # Allow up to 5 words on the last line if it's the final chunk
                if i + 3 >= len(words):
                    chunk = words[i:i+5]

                start_time = segment_start_time + i * time_per_word
                end_time = start_time + len(chunk) * time_per_word

                # Format words with emphasis on the last word
                formatted_words = []
                for j, word in enumerate(chunk):
                    if j == len(chunk) - 1:
                        formatted_words.append(f"<font color=\"#ffff00\">{word}</font>")  # Yellow for emphasis
                    else:
                        formatted_words.append(word)

                f.write(f"{subtitle_index}\n")
                f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                f.write(f"{' '.join(formatted_words)}\n\n")
                subtitle_index += 1

def create_organized_clips(json_file, video_file, base_output_dir, transcript_path):
    """Create and organize video clips from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        clips_data = json.load(f)

    # Create base output directory
    base_dir = Path(base_output_dir)
    clips_dir = base_dir / "processed_clips"
    clips_dir.mkdir(exist_ok=True, parents=True)

    for i, clip in enumerate(clips_data, 1):
        try:
            # Generate output paths
            clip_name = f"clip_{i:03d}"
            video_output = clips_dir / f"{clip_name}.mp4"
            subtitle_output = clips_dir / f"{clip_name}.srt"

            # Get transcript segments for this clip
            segments = get_transcript_segments(
                transcript_path,
                float(clip['start_time']),
                float(clip['end_time'])
            )

            # Create subtitle file
            create_subtitle_file(segments, subtitle_output)

            # Ensure subtitle file exists before running ffmpeg
            if not subtitle_output.exists():
                print(f"Error: Subtitle file not created: {subtitle_output}")
                continue

            # Use absolute paths for ffmpeg command
            video_file = Path(video_file).resolve()
            video_output = video_output.resolve()
            subtitle_output = subtitle_output.resolve()

            # Create video clip
            create_video_clips_util(
                [clip],
                str(video_file),
                str(video_output.parent),
                str(transcript_path),
                output_filename=video_output.name
            )

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error for clip {i}:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Error output: {e.stderr}")
        except Exception as e:
            print(f"Error creating clip {i}: {str(e)}")
            import traceback
            print(traceback.format_exc())

def create_video_clips_util(json_data, video_file, output_dir, transcript_path, zoom_factor=1.1, x_pan=0, y_pan=0, output_filename=None):
    """Create vertical video clips (9:16) with subtitles, zoom, and pan."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get video dimensions
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
                                str(video_file)],
                               capture_output=True, text=True, check=True)
        width, height = map(int, probe.stdout.strip().split('x'))

        # Calculate cropping parameters for 9:16 aspect ratio
        target_width = height * 9 // 16
        crop_x = (width - target_width) // 2 if width > target_width else 0
        crop_width = target_width if width > target_width else width

        # Define zoom and pan animation
        zoom_animation = f'zoompan=z=\'min(zoom+0.0005,1.2)\':x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':d=1:s=1080x1920'

        # Process each clip
        for i, clip in enumerate(json_data, 1):
            start_time = float(clip['start_time'])
            end_time = float(clip['end_time'])
            reason = clip.get('reason', '')

            # Create output filename
            if output_filename:
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_filename = f"clip_{i:02d}_{int(start_time)}_{int(end_time)}.mp4"
                output_path = os.path.join(output_dir, output_filename)

            # Get transcript segments for this clip
            segments = get_transcript_segments(transcript_path, start_time, end_time)

            # Create subtitle file with improved formatting
            subs_filename = f"clip_{i:02d}_subs.srt" if not output_filename else f"{Path(output_filename).stem}_subs.srt"
            subs_path = os.path.join(output_dir, subs_filename)
            create_subtitle_file(segments, subs_path)

            print(f"\nCreating clip {i}: {output_filename}")
            print(f"Timestamp: {start_time}s to {end_time}s")
            print(f"Reason: {reason}")

            # FFmpeg command with 9:16 aspect ratio, zoom, and pan
            duration = end_time - start_time

            # Correctly escape the subtitle path for FFmpeg
            subs_path_escaped = subs_path.replace("\\", "\\\\").replace(":", "\\:").replace(",", "\\,")

            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', video_file,
                '-vf',
                f'crop={crop_width}:{height}:{crop_x}:0,'
                f'scale=1080:1920,setsar=1,'  # Scale to 1080:1920 for better quality (TikTok's max resolution)
                f'{zoom_animation},'
                f'pad=iw:ih+80:0:40:color=black,'  # Add black padding for subtitles
                f'subtitles=\'{subs_path_escaped}\'',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',  # Higher quality
                '-c:a', 'aac',
                '-b:a', '192k',  # Higher audio bitrate
                output_path
            ]

            # Execute FFmpeg command
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"Successfully created clip: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error creating clip {i}:")
                print(f"Command: {' '.join(cmd)}")
                print(e.stderr)

    except Exception as e:
        print(f"Error in create_video_clips: {str(e)}")

# --- Main Function ---

def main():
    # Setup paths
    base_dir = Path("downloads")
    clips_dir = base_dir / "clips"
    videos_dir = base_dir / "videos"

    # Ensure directories exist
    for dir_path in [base_dir, clips_dir, videos_dir]:
        dir_path.mkdir(exist_ok=True)

    transcript_files = list(TRANSCRIPTS_DIR.glob("*_transcript.txt"))
    
    if not transcript_files:
        print("\nNo transcript files found in the transcripts directory.")
        return

    # Get the most recently created transcript file
    latest_transcript = max(transcript_files, key=lambda x: x.stat().st_mtime)
    print(f"\nAnalyzing latest transcript: {latest_transcript.name}")
    

    # Get the video file and transcript path
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        print("No video files found in downloads/videos directory")
        exit(1)

    video_file = video_files[0]  # Use the first video file found
    transcript_path = base_dir / "transcripts" / f"{video_file.stem}_transcript.txt"

    if not transcript_path.exists():
        print(f"Transcript file not found: {transcript_path}")
        exit(1)
    
    # Find the corresponding _clips.json file
    clips_json_file = CLIPS_DIR / f"{video_file.stem}_clips.json"

    if not clips_json_file.exists():
        print(f"Clips JSON file not found: {clips_json_file}. Running analysis...")
        analyze_transcript_with_llm(transcript_path) # Analyze transcript and generate clips JSON

    # Create and organize clips
    print("\nCreating organized clips...")
    create_organized_clips(clips_json_file, video_file, clips_dir, transcript_path)
    print("\nDone! Check the clips directory for the processed clips.")


if __name__ == "__main__":
    initialize_google_api()
    main()