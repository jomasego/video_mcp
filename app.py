import gradio as gr
import os
import requests
import tempfile
import subprocess
import re
import shutil # Added for rmtree
import modal
from typing import Dict, Any # Added for type hinting

def is_youtube_url(url_string: str) -> bool:
    """Checks if the given string is a YouTube URL."""
    # More robust regex to find YouTube video ID, accommodating various URL formats
    # and additional query parameters.
    youtube_regex = (
        r'(?:youtube(?:-nocookie)?\.com/(?:[^/\n\s]+/|watch(?:/|\?(?:[^&\n\s]+&)*v=)|embed(?:/|\?(?:[^&\n\s]+&)*feature=oembed)|shorts/|live/)|youtu\.be/)'
        r'([a-zA-Z0-9_-]{11})' # This captures the 11-character video ID
    )
    # We use re.search because the video ID might not be at the start of the query string part of the URL.
    # re.match only matches at the beginning of the string (or beginning of line in multiline mode).
    # The regex now directly looks for the 'v=VIDEO_ID' or youtu.be/VIDEO_ID structure.
    # The first part of the regex matches the domain and common paths, the second part captures the ID.
    return bool(re.search(youtube_regex, url_string))

def download_video(url_string: str, temp_dir: str) -> str | None:
    """Downloads video from a URL (YouTube or direct link) to a temporary directory."""
    if is_youtube_url(url_string):
        print(f"Attempting to download YouTube video: {url_string}")
        # Define a fixed output filename pattern within the temp_dir
        output_filename_template = "downloaded_video.%(ext)s" # yt-dlp replaces %(ext)s
        output_path_template = os.path.join(temp_dir, output_filename_template)

        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best", # Prefer mp4 format
            "--output", output_path_template,
            url_string
        ]
        print(f"Executing yt-dlp command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

            print(f"yt-dlp STDOUT:\n{result.stdout}")
            print(f"yt-dlp STDERR:\n{result.stderr}")

            if result.returncode == 0:
                # Find the actual downloaded file based on the template
                downloaded_file_path = None
                for item in os.listdir(temp_dir):
                    if item.startswith("downloaded_video."):
                        potential_path = os.path.join(temp_dir, item)
                        if os.path.isfile(potential_path):
                            downloaded_file_path = potential_path
                            print(f"YouTube video successfully downloaded to: {downloaded_file_path}")
                            break
                if downloaded_file_path:
                    return downloaded_file_path
                else:
                    print(f"yt-dlp seemed to succeed (exit code 0) but the output file 'downloaded_video.*' was not found in {temp_dir}.")
                    return None
            else:
                print(f"yt-dlp failed with return code {result.returncode}.")
                return None
        except subprocess.TimeoutExpired:
            print(f"yt-dlp command timed out after 300 seconds for URL: {url_string}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during yt-dlp execution for {url_string}: {e}")
            return None

    elif url_string.startswith(('http://', 'https://')) and url_string.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        print(f"Attempting to download direct video link: {url_string}")
        try:
            response = requests.get(url_string, stream=True, timeout=300) # 5 min timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            
            filename = os.path.basename(url_string) or "downloaded_video_direct.mp4"
            video_file_path = os.path.join(temp_dir, filename)
            
            with open(video_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Direct video downloaded successfully to: {video_file_path}")
            return video_file_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading direct video link {url_string}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during direct video download for {url_string}: {e}")
            return None
    else:
        print(f"Input '{url_string}' is not a recognized YouTube URL or direct video link for download.")
        return None


def process_video_input(input_string: str) -> Dict[str, Any]:
    """
    Processes the video (from URL or local file path) and returns its transcription status as a JSON object.
    """
    if not input_string:
        return {
            "status": "error",
            "error_details": {
                "message": "No video URL or file path provided.",
                "input_received": input_string
            }
        }

    video_path_to_process = None
    created_temp_dir = None  # To store path of temp directory if created for download

    try:
        if input_string.startswith(('http://', 'https://')):
            print(f"Input is a URL: {input_string}")
            created_temp_dir = tempfile.mkdtemp()
            print(f"Created temporary directory for download: {created_temp_dir}")
            downloaded_path = download_video(input_string, created_temp_dir)
            
            if downloaded_path and os.path.exists(downloaded_path):
                video_path_to_process = downloaded_path
            else:
                # Error message is already printed by download_video or this block
                print(f"Failed to download or locate video from URL: {input_string}")
                # Cleanup is handled in finally, so just return error
                return {
                    "status": "error",
                    "error_details": {
                        "message": "Failed to download video from URL.",
                        "input_received": input_string
                    }
                }
        
        elif os.path.exists(input_string):
            print(f"Input is a local file path: {input_string}")
            video_path_to_process = input_string
        else:
            return {
                "status": "error",
                "error_details": {
                    "message": f"Input '{input_string}' is not a valid URL or an existing file path.",
                    "input_received": input_string
                }
            }

        if video_path_to_process:
            print(f"Processing video: {video_path_to_process}")
            print(f"Video path to process: {video_path_to_process}")
            try:
                print("Reading video file into bytes...")
                with open(video_path_to_process, "rb") as video_file:
                    video_bytes_content = video_file.read()
                print(f"Read {len(video_bytes_content)} bytes from video file.")

                # Ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set as environment variables
                # in your Hugging Face Space. For local `python app.py` runs, Modal CLI's
                # authenticated state is usually used.
                # os.environ["MODAL_TOKEN_ID"] = "your_modal_token_id" # Replace or set in HF Space
                # os.environ["MODAL_TOKEN_SECRET"] = "your_modal_token_secret" # Replace or set in HF Space

                print("Looking up Modal function 'whisper-transcriber/transcribe_video_audio'...")
                # The function name should match what was deployed. 
                # It's typically 'AppName/FunctionName' or just 'FunctionName' if app is default.
                # Based on your deployment log, app name is 'whisper-transcriber'
                # and function is 'transcribe_video_audio'
                try:
                    f = modal.Function.from_name("whisper-transcriber", "transcribe_video_audio")
                    print("Modal function looked up successfully.")
                except modal.Error as e:
                    print("Modal function 'whisper-transcriber/transcribe_video_audio' not found. Trying with just function name.")
                    # Fallback or alternative lookup, though the above should be correct for named apps.
                    # This might be needed if the app name context is implicit.
                    # For a named app 'whisper-transcriber' and function 'transcribe_video_audio',
                    # the lookup `modal.Function.lookup("whisper-transcriber", "transcribe_video_audio")` is standard.
                    # If it was deployed as part of the default app, then just "transcribe_video_audio" might work.
                    # Given the deployment log, the first lookup should be correct.
                    return {
                        "status": "error",
                        "error_details": {
                            "message": "Could not find the deployed Modal function. Please check deployment status and name.",
                            "modal_function_name": "whisper-transcriber/transcribe_video_audio"
                        }
                    }

                print("Calling Modal function for transcription...")
                # Using .remote() for asynchronous execution, .call() for synchronous
                # For Gradio, synchronous (.call()) might be simpler to handle the response directly.
                transcription = f.remote(video_bytes_content) # Use .remote() for Modal function call
                print(f"Received transcription from Modal: {transcription[:100]}...")
                return {
                    "status": "success",
                    "data": {
                        "transcription": transcription
                    }
                }
            except FileNotFoundError:
                print(f"Error: Video file not found at {video_path_to_process} before sending to Modal.")
                return {
                    "status": "error",
                    "error_details": {
                        "message": "Video file disappeared before processing.",
                        "path_attempted": video_path_to_process
                    }
                }
            except modal.Error as e: # Using modal.Error as the base Modal exception
                print(f"Modal specific error: {e}")
                return {
                    "status": "error",
                    "error_details": {
                        "message": f"Error during Modal operation: {str(e)}",
                        "exception_type": type(e).__name__
                    }
                }
            except Exception as e:
                print(f"An unexpected error occurred while calling Modal: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "error_details": {
                        "message": f"Failed to get transcription: {str(e)}",
                        "exception_type": type(e).__name__
                    }
                }
        else:
            # This case should ideally be caught by earlier checks
            return {
                "status": "error",
                "error_details": {
                    "message": "No video available to process after input handling.",
                    "input_received": input_string
                }
            }
            
    finally:
        if created_temp_dir and os.path.exists(created_temp_dir):
            print(f"Cleaning up temporary directory: {created_temp_dir}")
            try:
                shutil.rmtree(created_temp_dir)
                print(f"Successfully removed temporary directory: {created_temp_dir}")
            except Exception as e:
                print(f"Error removing temporary directory {created_temp_dir}: {e}")

# Gradio Interface for the API endpoint
api_interface = gr.Interface(
    fn=process_video_input,
    inputs=gr.Textbox(lines=1, label="Video URL or Local File Path for Interpretation",
                      placeholder="Enter YouTube URL, direct video URL (.mp4, .mov, etc.), or local file path..."),
    outputs=gr.JSON(label="API Response"),
    title="Video Interpretation Input",
    description="Provide a video URL or local file path to get its interpretation status as JSON.",
    allow_flagging="never",
    examples=[
        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        ["https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"]
    ]
)

# Gradio Interface for a simple user-facing demo
def demo_process_video(input_string: str) -> str:
    """
    A simple demo function for the Gradio UI.
    It calls the same backend logic as the API.
    """
    print(f"Demo received input: {input_string}")
    result = process_video_input(input_string) # Call the core logic
    return result

demo_interface = gr.Interface(
    fn=demo_process_video,
    inputs=gr.Textbox(label="Upload Video URL or Local File Path for Demo", 
                      placeholder="Enter YouTube URL, direct video URL (.mp4, .mov, etc.), or local file path..."),
    outputs="text",
    title="Video Interpretation Demo",
    description="Provide a video URL or local file path to see its transcription status.",
    allow_flagging="never"
)

# JavaScript to find and replace the 'Use via API' link text
# This attempts to change the text a few times in case Gradio renders elements late.
js_code_for_head = """
(function() {
    console.log('[MCP Script] Initializing script to change API link text...');
    let foundAndChangedGlobal = false; // Declare here to be accessible in setInterval

    function attemptChangeApiLinkText() {
        const links = document.querySelectorAll('a');
        // console.log('[MCP Script] Found ' + links.length + ' anchor tags on this attempt.');
        for (let i = 0; i < links.length; i++) {
            const linkText = links[i].textContent ? links[i].textContent.trim() : '';
            if (linkText === 'Use via API' || linkText === 'Share via Link') { // Target both possible texts
                links[i].textContent = 'Use as an MCP or via API';
                console.log('[MCP Script] Successfully changed link text from: ' + linkText);
                foundAndChangedGlobal = true;
                return true; // Indicate success
            }
        }
        return false; // Indicate not found/changed in this attempt
    }

    let attempts = 0;
    const maxAttempts = 50; // Try for up to 5 seconds (50 * 100ms)
    let initialScanDone = false;

    const intervalId = setInterval(() => {
        if (!initialScanDone && attempts === 0) {
            console.log('[MCP Script] Performing initial scan for API link text.');
            initialScanDone = true;
        }

        if (attemptChangeApiLinkText() || attempts >= maxAttempts) {
            clearInterval(intervalId);
            if (attempts >= maxAttempts && !foundAndChangedGlobal) {
                console.log('[MCP Script] Max attempts reached. Target link was not found or changed. It might not be rendered or has a different initial text.');
            }
        }
        attempts++;
    }, 100);
})();
"""

# Combine interfaces into a Blocks app
with gr.Blocks(head=f"<script>{js_code_for_head}</script>") as app:
    gr.Markdown("# LLM Video interpretation MCP")
    gr.Markdown("This Hugging Face Space acts as a backend for processing video context for AI models.")

    with gr.Tab("API Endpoint (for AI Models)"):
        gr.Markdown("### Use this endpoint from another application (e.g., another Hugging Face Space).")
        gr.Markdown("The `process_video_input` function (for video interpretation) is exposed here.")
        api_interface.render()
        gr.Markdown("**Note:** Some YouTube videos may fail to download if they require login or cookie authentication due to YouTube's restrictions. Direct video links are generally more reliable for automated processing.")

    with gr.Tab("Demo (for Manual Testing)"):
        gr.Markdown("### Manually test video URLs or paths for interpretation and observe the JSON response.")
        demo_interface.render()

# Launch the Gradio application
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
