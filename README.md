# Video MCP Transcription Service

This project provides a video transcription service using Python, Modal, and Hugging Face's Whisper model. It allows users to submit a video URL (e.g., YouTube) or upload a video file, and it returns the transcribed audio text.

## Features

-   Downloads videos from URLs using `yt-dlp`.
-   Extracts audio and transcribes it using a Modal-deployed Whisper model.
-   Provides a simple web interface using Gradio for local testing.
-   The Modal function (`modal_whisper_app.py`) handles the heavy lifting of transcription in a serverless environment.

## Project Structure

-   `app.py`: The main Gradio application for local interaction and calling the Modal function.
-   `modal_whisper_app.py`: Defines the Modal app and the `transcribe_video_audio` function that runs the Whisper model.
-   `requirements.txt`: Lists Python dependencies for the local `app.py`.

## Setup

### Prerequisites

-   Python 3.10+
-   Modal account and CLI installed and configured (`pip install modal-client`, then `modal setup`).
-   `ffmpeg` installed locally (for `yt-dlp` and `moviepy` to process video/audio).
    -   On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    -   On macOS (using Homebrew): `brew install ffmpeg`

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jomasego/video_mcp.git
    cd video_mcp
    ```

2.  **Install local dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Deploy the Modal function (if not already deployed or if changes were made):**
    Ensure your Modal CLI is authenticated.
    ```bash
    modal deploy modal_whisper_app.py
    ```
    This deploys the `transcribe_video_audio` function to Modal. You should see a success message with a deployment URL.

### Running the Local Application

1.  **Start the Gradio app:**
    ```bash
    python3 app.py
    ```
2.  Open your web browser and go to the URL provided by Gradio (usually `http://127.0.0.1:7860`).
3.  Enter a video URL or upload a video file to get the transcription.

## Modal Function Details

The `modal_whisper_app.py` script defines a Modal function that:
-   Uses a custom Docker image with `ffmpeg`, `transformers`, `torch`, `moviepy`, `soundfile`, and `huggingface_hub`.
-   Takes video bytes as input.
-   Uses `moviepy` to extract audio from the video.
-   Uses the Hugging Face `transformers` pipeline with a specified Whisper model (e.g., `openai/whisper-large-v3`) to transcribe the audio.
-   Requires a Hugging Face token stored as a Modal secret (`HF_TOKEN_SECRET`) if using gated models or for authenticated access.

## Future Work

-   Deploy as an MCP (Multi-Compute Platform) server on Hugging Face Spaces.
-   Develop a chat interface (e.g., using Claude 3.5 Sonnet) to interact with the transcription service, allowing users to ask questions about the video content based on the transcription.

## Troubleshooting

-   **`ModuleNotFoundError: No module named 'moviepy.editor'` (in Modal logs):**
    This indicates `moviepy` might not be correctly installed in the Modal image. Ensure `moviepy` is in `pip_install` and/or `run_commands("pip install moviepy")` in `modal_whisper_app.py` and redeploy.
-   **`yt-dlp` errors or warnings about `ffmpeg`:**
    Ensure `ffmpeg` is installed on your local system where `app.py` is run, and also within the Modal image (`apt_install("ffmpeg")`).
-   **Modal authentication errors:**
    Ensure `modal setup` has been run and your Modal token is active. For Hugging Face Spaces, Modal tokens might need to be set as environment variables/secrets.
