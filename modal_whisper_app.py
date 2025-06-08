import modal
import os
import tempfile
import io

# Environment variable for model name, configurable in Modal UI or via .env
# This will be used by both the pre-caching function and the runtime function
WHISPER_MODEL_NAME = os.environ.get("HF_WHISPER_MODEL_NAME", "openai/whisper-large-v3")
CAPTION_MODEL_NAME = "Neleac/SpaceTimeGPT"
CAPTION_PROCESSOR_NAME = "MCG-NJU/videomae-base"
CAPTION_TOKENIZER_NAME = "gpt2" # SpaceTimeGPT uses GPT-2 as decoder
ACTION_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
ACTION_PROCESSOR_NAME = "MCG-NJU/videomae-base-finetuned-kinetics" # Often the same as model for VideoMAE

# Initialize a Modal Dict for caching results
# The key will be a hash of the video URL or video content
video_analysis_cache = modal.Dict.from_name(
    "video-analysis-cache", create_if_missing=True
)

def download_whisper_model():
    import torch
    from transformers import pipeline
    print(f"Downloading and caching Whisper model: {WHISPER_MODEL_NAME}")
    pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        torch_dtype=torch.float32,
        device="cpu"
    )
    print(f"Whisper model {WHISPER_MODEL_NAME} cached successfully.")

def download_caption_model():
    import torch
    from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
    print(f"Downloading and caching caption model: {CAPTION_MODEL_NAME}")
    # Download image processor
    AutoImageProcessor.from_pretrained(CAPTION_PROCESSOR_NAME)
    print(f"Image processor {CAPTION_PROCESSOR_NAME} cached.")
    # Download tokenizer
    AutoTokenizer.from_pretrained(CAPTION_TOKENIZER_NAME)
    print(f"Tokenizer {CAPTION_TOKENIZER_NAME} cached.")
    # Download main model
    VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL_NAME)
    print(f"Caption model {CAPTION_MODEL_NAME} cached successfully.")

def download_action_model():
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    print(f"Downloading and caching action recognition model: {ACTION_MODEL_NAME}")
    # Download image processor
    VideoMAEImageProcessor.from_pretrained(ACTION_PROCESSOR_NAME)
    print(f"Action model processor {ACTION_PROCESSOR_NAME} cached.")
    # Download main model
    VideoMAEForVideoClassification.from_pretrained(ACTION_MODEL_NAME)
    print(f"Action model {ACTION_MODEL_NAME} cached successfully.")

# Define the Modal image
whisper_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .run_commands(
        "echo 'Force reinstalling moviepy...'",
        "pip install --force-reinstall moviepy",
        "echo 'Checking moviepy installation...'",
        "pip show moviepy || echo 'pip show moviepy failed'", 
        "echo 'Attempting to import moviepy.editor during build:'",
        "python -c 'import moviepy; print(f\"moviepy module loaded from: {moviepy.__file__}\"); from moviepy.video.io.VideoFileClip import VideoFileClip; print(\"moviepy.video.io.VideoFileClip.VideoFileClip class import successful\")'"
    )  # Force install moviepy and add diagnostics
    .pip_install(
        "transformers[torch]",
        "accelerate",
        "soundfile",
        "moviepy",  # Essential for audio extraction from video
        "huggingface_hub",
        "ffmpeg-python",
        "av",  # For video frame extraction
        "fastapi[standard]" # For web endpoints
    )
    .run_function(download_whisper_model)
    .run_function(download_caption_model)
    .run_function(download_action_model) # This runs download_action_model during image build
)

app = modal.App(name="whisper-transcriber") # Changed from modal.Stub to modal.App



# Hugging Face Token - retrieve from memory and set as Modal Secret
# IMPORTANT: Create a Modal Secret named 'my-huggingface-secret' with your actual HF_TOKEN.
# Example: modal secret create my-huggingface-secret HF_TOKEN=your_hf_token_here
HF_TOKEN_SECRET = modal.Secret.from_name("my-huggingface-secret")

@app.function(
    image=whisper_image, 
    secrets=[HF_TOKEN_SECRET],
    timeout=1200,
    gpu="any"  # Request any available GPU
)
def transcribe_video_audio(video_bytes: bytes) -> str:
    # Imports moved inside the function to avoid local ModuleNotFoundError during `modal deploy`
    from moviepy.video.io.VideoFileClip import VideoFileClip # More specific import for moviepy 2.2.1
    import soundfile as sf
    import torch
    from transformers import pipeline # This will now use the pre-cached model
    from huggingface_hub import login

    if not video_bytes:
        return "Error: No video data received."

    # Login to Hugging Face Hub using the token from Modal secrets
    hf_token = os.environ.get("HF_TOKEN") # Standard key for Hugging Face token in Modal secrets if set as HF_TOKEN=...
    if hf_token:
        try:
            login(token=hf_token)
            print("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"Hugging Face Hub login failed: {e}. Proceeding, but private models may not be accessible.")
    else:
        print("HF_TOKEN secret not found. Proceeding without login (works for public models).")

    print(f"Processing video for transcription using model: {WHISPER_MODEL_NAME}")
    
    # Initialize pipeline inside the function.
    # For production/frequent use, consider @stub.cls to load the model once per container lifecycle.
    print("Loading Whisper model...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Use float16 for GPU for faster inference and less memory, float32 for CPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        torch_dtype=torch_dtype,
        device=device_map,
    )
    print(f"Whisper model loaded on device: {device_map} with dtype: {torch_dtype}")

    video_path = None
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
            tmp_video_file.write(video_bytes)
            video_path = tmp_video_file.name
        print(f"Temporary video file saved: {video_path}")

        print("Extracting audio from video...")
        video_clip = VideoFileClip(video_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            audio_path = tmp_audio_file.name
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None) 
        video_clip.close()
        print(f"Audio extracted to: {audio_path}")

        audio_input, samplerate = sf.read(audio_path)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(axis=1) # Convert to mono
        
        print(f"Audio data shape: {audio_input.shape}, Samplerate: {samplerate}")
        print("Starting transcription...")
        # Pass audio as a dictionary for more control, or directly as numpy array
        # Adding chunk_length_s for handling long audio files better.
        result = transcriber(audio_input.copy(), chunk_length_s=30, batch_size=8, return_timestamps=False, generate_kwargs={"temperature": 0.2, "no_repeat_ngram_size": 3, "language": "en"})
        transcribed_text = result["text"]
        
        print(f"Transcription successful. Length: {len(transcribed_text)}")
        if len(transcribed_text) > 100:
            print(f"Transcription preview: {transcribed_text[:100]}...")
        else:
            print(f"Transcription: {transcribed_text}")
            
        return transcribed_text

    except Exception as e:
        print(f"Error during transcription process: {e}")
        import traceback
        traceback.print_exc() 
        return f"Error: Transcription failed. Details: {str(e)}"
    finally:
        for p in [video_path, audio_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"Removed temporary file: {p}")
                except Exception as e_rm:
                    print(f"Error removing temporary file {p}: {e_rm}")

# This is a local entrypoint for testing the Modal function if you run `modal run modal_whisper_app.py`
@app.local_entrypoint()
def main():
    # This is just an example of how you might test. 
    # You'd need a sample video file (e.g., "sample.mp4") in the same directory.
    # For actual deployment, this main function isn't strictly necessary as Gradio will call the webhook.
    sample_video_path = "sample.mp4" 
    if not os.path.exists(sample_video_path):
        print(f"Sample video {sample_video_path} not found. Skipping local test run.")
        return

    with open(sample_video_path, "rb") as f:
        video_bytes_content = f.read()
    
    print(f"Testing transcription with {sample_video_path}...")
    transcription = transcribe_video_audio.remote(video_bytes_content)
    print("----")
    print(f"Transcription Result: {transcription}")
    print("----")

# To call this function from another Python script (after deployment):
# import modal
# Ensure the app name matches the one in modal.App(name=...)
# The exact lookup method might vary slightly with modal.App, often it's:
# deployed_app = modal.App.lookup("whisper-transcriber") 
# or by accessing the function directly if the app is deployed with a name.
# For a deployed function, you might use its tag or webhook URL directly.
# Example using a direct function call if deployed and accessible:
# f = modal.Function.lookup("whisper-transcriber/transcribe_video_audio") # Or similar based on deployment output
# For invoking: 
# result = f.remote(your_video_bytes) # for async
# print(result)
# Or, if you have the app object:
# result = app.functions.transcribe_video_audio.remote(your_video_bytes)
# Consult Modal documentation for the precise invocation method for your Modal version and deployment style.

# Note: When deploying to Modal, Modal uses the `app.serve()` or `app.deploy()` mechanism.
# The Gradio app will call the deployed Modal function via its HTTP endpoint.

@app.function(
    image=whisper_image,
    secrets=[HF_TOKEN_SECRET],
    timeout=900, # Potentially shorter if model is pre-loaded and efficient
    gpu="any" # Request any available GPU
)
def generate_video_caption(video_bytes: bytes) -> str:
    import torch
    import av # PyAV for frame extraction
    from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
    import tempfile
    import os
    import numpy as np

    if not video_bytes:
        return "Error: No video data received for captioning."

    print(f"Starting video captioning with {CAPTION_MODEL_NAME}...")
    video_path = None
    try:
        # 1. Load pre-cached model, processor, and tokenizer
        # Ensure these names match what's used in download_caption_model
        image_processor = AutoImageProcessor.from_pretrained(CAPTION_PROCESSOR_NAME)
        tokenizer = AutoTokenizer.from_pretrained(CAPTION_TOKENIZER_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL_NAME)
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Caption model loaded on device: {device}")

        # 2. Save video_bytes to a temporary file to be read by PyAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
            tmp_video_file.write(video_bytes)
            video_path = tmp_video_file.name
        print(f"Temporary video file for captioning saved: {video_path}")

        # 3. Frame extraction using PyAV
        container = av.open(video_path)
        # Select 8 frames evenly spaced throughout the video
        # Similar to the SpaceTimeGPT example
        total_frames = container.streams.video[0].frames
        num_frames_to_sample = 8
        indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        
        frames = []
        container.seek(0) # Reset stream to the beginning
        frame_idx = 0
        target_idx_ptr = 0
        for frame in container.decode(video=0):
            if target_idx_ptr < len(indices) and frame_idx == indices[target_idx_ptr]:
                frames.append(frame.to_image()) # Convert to PIL Image
                target_idx_ptr += 1
            frame_idx += 1
            if len(frames) == num_frames_to_sample:
                break
        container.close()
        
        if not frames:
            print("No frames extracted, cannot generate caption.")
            return "Error: Could not extract frames for captioning."
        print(f"Extracted {len(frames)} frames for captioning.")

        # 4. Generate caption
        # The SpaceTimeGPT example doesn't use a specific prompt, it generates from frames directly
        pixel_values = image_processor(images=frames, return_tensors="pt").pixel_values.to(device)
        # The model card for Neleac/SpaceTimeGPT uses max_length=128, num_beams=5
        generated_ids = model.generate(pixel_values, max_length=128, num_beams=5)
        caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(f"Generated caption: {caption}")
        return caption

    except Exception as e:
        print(f"Error during video captioning: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: Video captioning failed. Details: {str(e)}"
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Removed temporary video file for captioning: {video_path}")
            except Exception as e_rm:
                print(f"Error removing temporary captioning video file {video_path}: {e_rm}")

@app.function(
    image=whisper_image,
    secrets=[HF_TOKEN_SECRET],
    timeout=1800, # Increased timeout for combined processing
    gpu="any"
)
@modal.concurrent(max_inputs=10) # Replaces allow_concurrent_inputs
@modal.fastapi_endpoint(method="POST") # Replaces web_endpoint
async def process_video_context(video_bytes: bytes, video_url: str = None):
    import json
    import hashlib

    if not video_bytes:
        return modal.Response(status_code=400, body=json.dumps({"error": "No video data provided."}))

    # Generate a cache key
    # If URL is provided, use it. Otherwise, hash the video content (can be slow for large videos).
    cache_key = ""
    if video_url:
        cache_key = hashlib.sha256(video_url.encode()).hexdigest()
    else:
        # Hashing large video_bytes can be memory/CPU intensive. Consider alternatives if this is an issue.
        # For now, let's proceed with hashing bytes if no URL.
        cache_key = hashlib.sha256(video_bytes).hexdigest()
    
    print(f"Generated cache key: {cache_key}")

    # Check cache first
    if cache_key in video_analysis_cache:
        print(f"Cache hit for key: {cache_key}")
        cached_result = video_analysis_cache[cache_key]
        return modal.Response(status_code=200, body=json.dumps(cached_result))
    
    print(f"Cache miss for key: {cache_key}. Processing video...")

    results = {}
    error_messages = []

    # Call transcription and captioning in parallel
    transcription_future = transcribe_video_audio.spawn(video_bytes)
    caption_call = generate_video_caption.spawn(video_bytes)
    action_call = generate_action_labels.spawn(video_bytes) # Placeholder for now

    try:
        transcription_result = await transcription_future
        if transcription_result.startswith("Error:"):
            error_messages.append(f"Transcription: {transcription_result}")
            results["transcription"] = None
        else:
            results["transcription"] = transcription_result
    except Exception as e:
        print(f"Error in transcription task: {e}")
        error_messages.append(f"Transcription: Failed with exception - {str(e)}")
        results["transcription"] = None

    try:
        caption_result = await caption_call
        if caption_result.startswith("Error:"):
            error_messages.append(f"Captioning: {caption_result}")
            results["video_caption"] = None
        else:
            results["video_caption"] = caption_result
    except Exception as e:
        print(f"Error in captioning task: {e}")
        error_messages.append(f"Captioning: Failed with exception - {str(e)}")
        results["video_caption"] = None

    try:
        action_result = await action_call # action_result is a dict from generate_action_labels
        if action_result.get("error"):
            error_messages.append(f"Action recognition: {action_result.get('error')}")
            results["action_recognition"] = None
        else:
            results["action_recognition"] = action_result.get("actions", "No actions detected or error in result format")
    except Exception as e:
        print(f"Error in action recognition task: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Action recognition: Failed with exception - {str(e)}")
        results["action_recognition"] = None

    # TODO: Add calls for object detection here in the future
    results["object_detection"] = "(Object detection/tracking not yet implemented)"

    if error_messages:
        results["processing_errors"] = error_messages
        # Store partial results in cache even if there are errors
        video_analysis_cache[cache_key] = results 
        return modal.Response(status_code=207, body=json.dumps(results)) # 207 Multi-Status
    
    # Store successful full result in cache
    video_analysis_cache[cache_key] = results
    print(f"Successfully processed and cached results for key: {cache_key}")
    return modal.Response(status_code=200, body=json.dumps(results))

# Update local entrypoint to use the new main processing function if desired for testing
# For now, keeping it as is to test transcription independently if needed.

@app.function(
    image=whisper_image,
    secrets=[HF_TOKEN_SECRET],
    timeout=700, # Increased timeout slightly for model loading and inference
    gpu="any" # Requires GPU
)
def generate_action_labels(video_bytes: bytes) -> dict:
    import torch
    import av
    import numpy as np
    import tempfile
    import os
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    from huggingface_hub import login

    if not video_bytes:
        return {"actions": [], "error": "No video data received."}

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Action Recognition: Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"Action Recognition: Hugging Face Hub login failed: {e}.")
    else:
        print("Action Recognition: HF_TOKEN secret not found. Proceeding without login.")

    video_path = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Action Recognition: Loading model on device: {device}")
        
        processor = VideoMAEImageProcessor.from_pretrained(ACTION_PROCESSOR_NAME)
        model = VideoMAEForVideoClassification.from_pretrained(ACTION_MODEL_NAME)
        model.to(device)
        model.eval()
        print(f"Action Recognition: Model {ACTION_MODEL_NAME} and processor loaded.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
            tmp_video_file.write(video_bytes)
            video_path = tmp_video_file.name
        
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        num_frames_to_extract = 16
        total_frames = stream.frames
        if total_frames == 0:
            return {"actions": [], "error": "Video stream has no frames."}

        # Ensure we don't try to select more frames than available, especially for very short videos
        if total_frames < num_frames_to_extract:
            print(f"Warning: Video has only {total_frames} frames, less than desired {num_frames_to_extract}. Using all available frames.")
            num_frames_to_extract = total_frames
            if num_frames_to_extract == 0: # Double check after adjustment
                 return {"actions": [], "error": "Video stream has no frames after adjustment."}

        indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
        
        frames = []
        container.seek(0) # Reset stream to the beginning before decoding specific frames
        frame_idx_counter = 0
        target_idx_ptr = 0
        for frame in container.decode(video=0):
            if target_idx_ptr < len(indices) and frame_idx_counter == indices[target_idx_ptr]:
                frames.append(frame.to_image()) # Convert to PIL Image
                target_idx_ptr += 1
            frame_idx_counter += 1
            if target_idx_ptr == len(indices):
                break
        
        container.close()

        if not frames:
            return {"actions": [], "error": "Could not extract frames from video."}

        print(f"Action Recognition: Extracted {len(frames)} frames.")

        # Process frames and predict
        inputs = processor(frames, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        
        print(f"Action Recognition: Predicted action: {predicted_label}")
        return {"actions": [predicted_label], "error": None}

    except Exception as e:
        print(f"Error during action recognition: {e}")
        import traceback
        traceback.print_exc()
        return {"actions": [], "error": f"Action recognition failed: {str(e)}"}
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Removed temporary video file for action recognition: {video_path}")
            except Exception as e_rm:
                print(f"Error removing temporary action recognition video file {video_path}: {e_rm}")

