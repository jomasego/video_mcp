import modal
import os
import tempfile
import io

# Define the Modal image
whisper_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .run_commands("pip install moviepy")  # Force install moviepy
    .pip_install(
        "transformers[torch]",
        "accelerate",
        "soundfile",
        "moviepy",  # Essential for audio extraction from video
        "huggingface_hub",
        "ffmpeg-python"
    )
)

app = modal.App(name="whisper-transcriber") # Changed from modal.Stub to modal.App

# Environment variable for model name, configurable in Modal UI or via .env
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "openai/whisper-base")

# Hugging Face Token - retrieve from memory and set as Modal Secret
# IMPORTANT: Create a Modal Secret named 'my-huggingface-secret' with your actual HF_TOKEN.
# Example: modal secret create my-huggingface-secret HF_TOKEN=your_hf_token_here
HF_TOKEN_SECRET = modal.Secret.from_name("my-huggingface-secret")

@app.function(
    image=whisper_image, 
    secrets=[HF_TOKEN_SECRET],
    timeout=1200  
)
def transcribe_video_audio(video_bytes: bytes) -> str:
    # Imports moved inside the function to avoid local ModuleNotFoundError during `modal deploy`
    from moviepy.editor import VideoFileClip
    import soundfile as sf
    import torch
    from transformers import pipeline
    from huggingface_hub import login

    if not video_bytes:
        return "Error: No video data received."

    # Login to Hugging Face Hub using the token from Modal secrets
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"Hugging Face Hub login failed: {e}. Proceeding, but private models may not be accessible.")
    else:
        print("HF_TOKEN secret not found. Proceeding without login (works for public models).")

    print(f"Processing video for transcription using model: {MODEL_NAME}")
    
    # Initialize pipeline inside the function.
    # For production/frequent use, consider @stub.cls to load the model once per container lifecycle.
    print("Loading Whisper model...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Use float16 for GPU for faster inference and less memory, float32 for CPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
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
        result = transcriber(audio_input.copy(), chunk_length_s=30, batch_size=8, return_timestamps=False)
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
