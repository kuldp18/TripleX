"""
This script processes video files using the JoyCaption2 model.
It extracts frames at a specified FPS, generates captions for each frame using JoyCaption2,
and creates a composite caption summarizing the entire video.

Usage:
  python captioners/joycaption2_video.py --dir /path/to/videos [--fps 1.0] [--max_frames 30] [--output_dir /path/to/output]
"""

import argparse
import json
import logging
import os
import shutil
import sys

import cv2
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Constants
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
MODEL_PATH = "models/llama-joycaption-alpha-two-hf-llava"
VALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mpeg", ".wmv", ".flv", ".mpg", ".webm", ".3gpp"}


def load_instructions(instruction_arg):
    """
    Load instructions from either inline text or a file path.
    
    Args:
        instruction_arg (str): Either inline text or a path to a text file.
    
    Returns:
        str: The loaded instructions, or empty string if None.
    """
    if not instruction_arg:
        return ""
    
    # Check if it's a file path
    if os.path.isfile(instruction_arg):
        try:
            with open(instruction_arg, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"Error reading instructions file '{instruction_arg}': {e}")
            return ""
    
    # Otherwise, treat it as inline text
    return instruction_arg.strip()

# Ensure model is downloaded
if not os.path.exists(MODEL_PATH):
    logging.info("Model not found. Downloading...")
    os.makedirs("models", exist_ok=True)
    snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_PATH)
    logging.info("Download complete!")

# Load model and processor
logging.info("Loading JoyCaption2 model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
llava_model.eval()
logging.info("Model loaded successfully!")


def caption_frame(image, timestamp, additional_instructions=""):
    """
    Generate a caption for a single video frame.

    Args:
        image (PIL.Image): The frame image.
        timestamp (int): The timestamp in seconds.
        additional_instructions (str): Optional additional instructions to append to the prompt.

    Returns:
        str: The generated caption.
    """
    try:
        # Build the base prompt
        base_prompt = f"Write a detailed descriptive caption for this video frame at {timestamp} seconds. Include information about the scene, people, actions, lighting, and camera angle. Use plain, everyday language."
        
        # Append additional instructions if provided
        if additional_instructions:
            base_prompt += f"\n\nAdditional instructions:\n{additional_instructions}"
        
        # Build the conversation prompt
        convo = [
            {"role": "system", "content": "You are a helpful video frame captioner."},
            {"role": "user", "content": base_prompt},
        ]

        # Format conversation for LLaVA
        convo_string = processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(
            "cuda"
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # Ensure tensors are valid
        if (
            torch.isnan(inputs["pixel_values"]).any()
            or torch.isinf(inputs["pixel_values"]).any()
        ):
            logging.error("Error: Input tensor contains NaN or Inf values.")
            return ""

        # Generate caption
        with torch.no_grad():
            generate_ids = llava_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )[0]

        # Trim the prompt from output
        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

        # Decode and clean up caption
        caption = processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        return caption

    except Exception as e:
        logging.error(f"Error generating caption for frame at {timestamp}s: {e}")
        return ""


def generate_composite_caption(frames_data, additional_instructions=""):
    """
    Generate a composite caption from all frame captions.

    Args:
        frames_data (list): List of dicts with timestamp and caption.
        additional_instructions (str): Optional additional instructions to append to the prompt.

    Returns:
        str: The composite caption.
    """
    if not frames_data:
        return ""

    # Build a prompt that includes all frame captions
    frame_descriptions = []
    for frame in frames_data:
        frame_descriptions.append(f"At {frame['timestamp']}s: {frame['caption']}")

    combined_text = "\n".join(frame_descriptions)

    prompt = f"""Based on the following frame-by-frame descriptions of a video, create a single cohesive paragraph that describes the entire video sequence. Merge redundant descriptions and maintain a natural flow. Use plain, everyday language.

Frame descriptions:
{combined_text}

Provide a single paragraph describing the complete video:"""
    
    # Append additional instructions if provided
    if additional_instructions:
        prompt += f"\n\nAdditional instructions:\n{additional_instructions}"

    try:
        # Use a simple text prompt without images for the composite
        convo = [
            {"role": "system", "content": "You are a helpful video description assistant."},
            {"role": "user", "content": prompt},
        ]

        convo_string = processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # Create a dummy white image for text-only generation
        dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        inputs = processor(
            text=[convo_string], images=[dummy_image], return_tensors="pt"
        ).to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generate_ids = llava_model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.7,
                top_k=None,
                top_p=0.9,
            )[0]

        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

        composite_caption = processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        return composite_caption

    except Exception as e:
        logging.error(f"Error generating composite caption: {e}")
        # Fallback: just concatenate frame captions
        return " ".join([f["caption"] for f in frames_data])


def process_video(file_path, fps=1.0, max_frames=None, output_dir=None, 
                  frame_instructions="", composite_instructions=""):
    """
    Process a video file and generate captions.

    Args:
        file_path (str): Path to the video file.
        fps (float): Frames per second to sample.
        max_frames (int): Maximum number of frames to process.
        output_dir (str): Directory to move completed files.
        frame_instructions (str): Additional instructions for frame captioning.
        composite_instructions (str): Additional instructions for composite caption generation.
    """
    base_name = os.path.splitext(file_path)[0]
    output_json_filename = base_name + ".json"
    output_txt_filename = base_name + ".txt"

    # Check if outputs already exist
    if os.path.exists(output_json_filename) and os.path.exists(output_txt_filename):
        try:
            with open(output_json_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(output_txt_filename, "r", encoding="utf-8") as f:
                composite_text = f.read().strip()
            if data.get("composite_caption", "").strip() and composite_text:
                logging.info(f"Skipping {file_path} because captions already exist.")
                return
        except Exception as e:
            logging.warning(f"Error reading existing caption files for {file_path}: {e}")

    logging.info(f"Processing video file: {file_path}")

    # Open video
    cap = cv2.VideoCapture(file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 25  # fallback

    frame_interval = max(1, round(video_fps / fps))
    count = 0
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            timestamp = int(count / video_fps)

            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Generate caption for this frame
            caption = caption_frame(pil_image, timestamp, frame_instructions)

            if caption:
                frames_data.append({
                    "timestamp": timestamp,
                    "caption": caption
                })
                logging.info(f"Frame at {timestamp}s captioned: {caption[:100]}...")

            # Check max_frames limit
            if max_frames is not None and len(frames_data) >= max_frames:
                logging.info(f"Reached maximum number of frames ({max_frames}). Stopping.")
                break

        count += 1

    cap.release()

    if not frames_data:
        logging.warning(f"No frames were captioned for {file_path}")
        return

    # Generate composite caption
    logging.info("Generating composite caption...")
    composite_caption = generate_composite_caption(frames_data, composite_instructions)
    logging.info(f"Composite caption: {composite_caption}")

    # Save outputs
    output_data = {
        "frames": frames_data,
        "composite_caption": composite_caption
    }

    with open(output_json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"JSON output saved to {output_json_filename}")

    with open(output_txt_filename, "w", encoding="utf-8") as f:
        f.write(composite_caption)
    logging.info(f"Caption text saved to {output_txt_filename}")

    # Move files to output directory if specified
    if output_dir and composite_caption.strip():
        try:
            os.makedirs(output_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(output_dir, os.path.basename(file_path)))
            shutil.move(output_json_filename, os.path.join(output_dir, os.path.basename(output_json_filename)))
            shutil.move(output_txt_filename, os.path.join(output_dir, os.path.basename(output_txt_filename)))
            logging.info(f"Moved video and output files to {output_dir}")
        except Exception as e:
            logging.error(f"Error moving files to output directory: {e}")


def process_directory(directory_path, fps=1.0, max_frames=None, output_dir=None,
                     frame_instructions="", composite_instructions=""):
    """
    Process all video files in a directory.

    Args:
        directory_path (str): Path to the directory containing videos.
        fps (float): Frames per second to sample.
        max_frames (int): Maximum number of frames to process per video.
        output_dir (str): Directory to move completed files.
        frame_instructions (str): Additional instructions for frame captioning.
        composite_instructions (str): Additional instructions for composite caption generation.
    """
    if not os.path.exists(directory_path):
        logging.error(f"Error: Directory '{directory_path}' not found!")
        return

    video_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
        and os.path.splitext(f)[1].lower() in VALID_VIDEO_EXTENSIONS
    ]

    if not video_files:
        logging.warning(f"No valid video files found in '{directory_path}'!")
        return

    logging.info(f"Found {len(video_files)} video files to process in '{directory_path}'...\n")

    for video_file in video_files:
        try:
            process_video(video_file, fps, max_frames, output_dir, 
                        frame_instructions, composite_instructions)
        except Exception as e:
            logging.error(f"Error processing {video_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for video files using JoyCaption2 model. "
                    "Extracts frames at specified FPS and creates both individual frame captions "
                    "and a composite caption for the entire video."
    )
    parser.add_argument("--dir", required=True, help="Directory containing video files")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to sample from videos (default: 1.0)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to caption per video (optional)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to move source files and captions after processing")
    parser.add_argument("--frame_instructions", type=str, default=None,
                        help="Additional instructions for frame captioning. Can be inline text or path to a text file.")
    parser.add_argument("--composite_instructions", type=str, default=None,
                        help="Additional instructions for composite caption generation. Can be inline text or path to a text file.")

    args = parser.parse_args()
    
    # Load instructions from inline text or files
    frame_instructions = load_instructions(args.frame_instructions)
    composite_instructions = load_instructions(args.composite_instructions)
    
    if frame_instructions:
        logging.info(f"Using frame instructions: {frame_instructions[:100]}...")
    if composite_instructions:
        logging.info(f"Using composite instructions: {composite_instructions[:100]}...")

    if os.path.isdir(args.dir):
        process_directory(args.dir, args.fps, args.max_frames, args.output_dir,
                        frame_instructions, composite_instructions)
    else:
        logging.error(f"Error: '{args.dir}' is not a valid directory!")
        sys.exit(1)


if __name__ == "__main__":
    main()
