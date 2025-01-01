import json
import re
from pathlib import Path
import subprocess
import os
import textwrap
import random

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
    pattern = r'\[([\d.]+)\s*-\s*([\d.]+)\]\s*(.+?)(?=\[|$)'
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
    """Create an SRT subtitle file with word-level timestamps, emphasis, and max 3 words per line."""
    with open(output_path, 'w', encoding='utf-8') as f:
        subtitle_index = 1
        for segment in segments:
            words = segment['text'].split()
            segment_start_time = segment['start']
            segment_duration = segment['end'] - segment['start']
            # Increase time per word slightly to match video pacing
            time_per_word = (segment_duration / len(words)) * 1.1 if len(words) > 0 else 0

            # Group words into chunks of up to 3 words
            for i in range(0, len(words), 3):
                chunk = words[i:i+3]
                start_time = segment_start_time + i * time_per_word
                # Add a small buffer between subtitle segments
                end_time = start_time + (len(chunk) * time_per_word) + 0.1

                # Format words with emphasis
                formatted_words = []
                # Randomly decide to highlight last 1 or 2 words
                num_highlight = random.randint(1, min(2, len(chunk)))
                highlight_start = len(chunk) - num_highlight

                for j, word in enumerate(chunk):
                    if j >= highlight_start:
                        formatted_words.append(f"<font color=\"#ffff00\">{word}</font>")  # Yellow for emphasis
                    else:
                        formatted_words.append(word)

                f.write(f"{subtitle_index}\n")
                f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                f.write(" ".join(formatted_words) + "\n\n")
                subtitle_index += 1

def process_google_responses(input_dir, output_json):
    """Process all Google response files into a single JSON file."""
    all_clips = []

    # Create output directory if it doesn't exist
    output_dir = Path(output_json).parent
    output_dir.mkdir(exist_ok=True)

    # Process all text files in the input directory
    for file_path in Path(input_dir).glob('google_response_*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find JSON content between triple backticks if present
            json_match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If not in backticks, try to find array directly
                json_match = re.search(r'(\[[\s\S]*?\])\s*$', content)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    continue

            # Clean the JSON string
            json_str = re.sub(r'^\s+|\s+$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)  # Remove comments

            # Parse clips from this file
            clips = json.loads(json_str)
            all_clips.extend(clips)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    if not all_clips:
        print("No valid clips found in any of the response files")
        # Create empty JSON file to avoid file not found error
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []

    # Write combined clips to output JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_clips, f, indent=4)

    return all_clips

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

        # Define zoom animation
        zoom_animation = f'zoompan=z=\'min(zoom+0.0005,1.2)\':d=1:s=1080x1920'

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

            # Create subtitle file with random yellow highlighting for last 1-2 words
            subs_filename = f"clip_{i:02d}_subs.srt" if not output_filename else f"{Path(output_filename).stem}_subs.srt"
            subs_path = os.path.join(output_dir, subs_filename)
            create_subtitle_file(segments, subs_path)

            print(f"\nCreating clip {i}: {output_filename}")
            print(f"Timestamp: {start_time}s to {end_time}s")
            print(f"Reason: {reason}")

            # Calculate the duration (now at normal speed)
            duration = end_time - start_time

            # Correctly escape the subtitle path and audio path for FFmpeg
            subs_path_escaped = subs_path.replace("\\", "\\\\").replace(":", "\\:").replace(",", "\\,")
            audio_path = os.path.join("downloads", "sound", "runaway.mp3").replace("\\", "/")

            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', video_file,
                '-stream_loop', '-1',  # Infinite input stream loop
                '-i', audio_path,  # Add background music
                '-filter_complex',
                f'[0:v]crop={crop_width}:{height}:{crop_x}:0,'
                f'scale=1080:1920,setsar=1,'  # Scale to 1080:1920
                f'colorbalance=rs=-1:gs=-1:bs=-1,' # Make it more black and white
                f'eq=contrast=1.2:brightness=0.1,' # Enhance contrast
                f'zoompan=z=\'min(zoom+0.0001,1.1)\':d=1:s=1080x1920:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\',' # Slower, smoother zoom
                f'pad=iw:ih+80:0:40:color=black[v1];'
                f'[v1]subtitles=\'{subs_path_escaped}\':force_style=\'FontName=Montserrat,FontSize=20,PrimaryColour=&Hffffff,SecondaryColour=&H00ffff,OutlineColour=&H000000,BackColour=&H66000000,Outline=2,BorderStyle=3,Shadow=1,MarginV=60,Alignment=2,Bold=1\'[v];'
                f'[1:a]volume=0.6[backsound];'  # Background music volume at 60%
                f'[0:a][backsound]amix=inputs=2:duration=longest[a]',  # Use longest duration
                '-map', '[v]',
                '-map', '[a]',
                '-shortest',  # End when the shortest input ends (the video)
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
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
            # Create category folder
            category = clip.get('category', 'uncategorized').lower()
            category_dir = clips_dir / category
            category_dir.mkdir(exist_ok=True, parents=True)

            # Generate output paths
            clip_name = f"clip_{i:03d}_{category}"
            video_output = category_dir / f"{clip_name}.mp4"
            subtitle_output = category_dir / f"{clip_name}.srt"

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

if __name__ == "__main__":
    # Setup paths
    base_dir = Path("downloads")
    google_responses_dir = base_dir / "deepseek_responses"
    clips_dir = base_dir / "clips"
    videos_dir = base_dir / "videos"
    combined_json = base_dir / "combined_clips.json"

    # Ensure directories exist
    for dir_path in [base_dir, google_responses_dir, clips_dir, videos_dir]:
        dir_path.mkdir(exist_ok=True)

    # Process all Google responses into a single JSON file
    print("Processing Google responses...")
    all_clips = process_google_responses(google_responses_dir, combined_json)
    print(f"Processed {len(all_clips)} clips into {combined_json}")

    # Get the video file and transcript path
    video_files = list(videos_dir.glob("*.mp4"))
    if not video_files:
        print("No video files found in downloads/videos directory")
        exit(1)

    video_file = video_files[0]  # Use the first video file found
    transcript_path = base_dir / "transcripts" / f"{video_file.stem}_transcript.txt"

    if not transcript_path.exists():
        print(f"Transcript file not found: {transcript_path}")
        exit(1)

    # Create and organize clips
    print("\nCreating organized clips...")
    create_organized_clips(combined_json, video_file, clips_dir, transcript_path)
    print("\nDone! Check the clips directory for the processed clips.")
