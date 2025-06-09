import gradio as gr
import os
import requests
import tempfile
import subprocess
import re
import shutil # Added for rmtree
import modal
from typing import Dict, Any, Optional # Added for type hinting

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

                print("Preparing to call Modal FastAPI endpoint for comprehensive analysis...")
                # IMPORTANT: Replace this with your actual Modal app's deployed FastAPI endpoint URL
                # This URL is typically found in your Modal deployment logs or dashboard.
                # It will look something like: https://YOUR_MODAL_WORKSPACE--video-analysis-gradio-pipeline-process-video-for-analysis.modal.run/analyze_video
                # Or, if the FastAPI endpoint function itself is not separately deployed but part of the main app deployment:
                # https://YOUR_MODAL_WORKSPACE--video-analysis-gradio-pipeline-fastapi-app.modal.run/analyze_video 
                # (The exact name depends on how Modal names the deployed web endpoint for the FastAPI app)
                # For now, using a placeholder. This MUST be configured.
                base_modal_url = os.getenv("MODAL_APP_BASE_URL")
                if not base_modal_url:
                    print("ERROR: MODAL_APP_BASE_URL environment variable not set.")
                    return {
                        "status": "error",
                        "error_details": {
                            "message": "Modal application base URL is not configured. Please set the MODAL_APP_BASE_URL environment variable.",
                            "input_received": input_string
                        }
                    }
                modal_endpoint_url = f"{base_modal_url.rstrip('/')}/analyze_video"

                files = {'video_file': (os.path.basename(video_path_to_process), video_bytes_content, 'video/mp4')}
                
                print(f"Calling Modal endpoint: {modal_endpoint_url}")
                try:
                    response = requests.post(modal_endpoint_url, files=files, timeout=1860) # Timeout slightly longer than Modal function
                    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                    analysis_results = response.json()
                    print(f"Received results from Modal endpoint: {str(analysis_results)[:200]}...")
                    return {
                        "status": "success",
                        "data": analysis_results
                    }
                except requests.exceptions.Timeout:
                    print(f"Request to Modal endpoint {MODAL_ENDPOINT_URL} timed out.")
                    return {
                        "status": "error",
                        "error_details": {
                            "message": "Request to video analysis service timed out.",
                            "endpoint_url": MODAL_ENDPOINT_URL
                        }
                    }
                except requests.exceptions.HTTPError as e:
                    print(f"HTTP error calling Modal endpoint {MODAL_ENDPOINT_URL}: {e.response.status_code} - {e.response.text}")
                    return {
                        "status": "error",
                        "error_details": {
                            "message": f"Video analysis service returned an error: {e.response.status_code}",
                            "details": e.response.text,
                            "endpoint_url": MODAL_ENDPOINT_URL
                        }
                    }
                except requests.exceptions.RequestException as e:
                    print(f"Error calling Modal endpoint {MODAL_ENDPOINT_URL}: {e}")
                    return {
                        "status": "error",
                        "error_details": {
                            "message": "Failed to connect to video analysis service.",
                            "details": str(e),
                            "endpoint_url": MODAL_ENDPOINT_URL
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
    flagging_options=None,
    examples=[
        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        ["https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"]
    ]
)

# Gradio Interface for a simple user-facing demo
def demo_process_video(input_string: str) -> tuple[str, Dict[str, Any]]:
    """
    A simple demo function for the Gradio UI.
    It calls process_video_input and unpacks its result for separate display.
    """
    result = process_video_input(input_string)
    status_str = result.get("status", "Unknown Status")
    
    # The second part of the tuple should be the 'data' if successful, 
    # or the 'error_details' (or the whole result) if there was an error.
    if status_str == "success" and "data" in result:
        details_json = result["data"]
    elif "error_details" in result:
        details_json = result["error_details"]
    else: # Fallback, show the whole result
        details_json = result
        
    return status_str, details_json


def call_topic_analysis_endpoint(topic_str: str, max_vids: int) -> Dict[str, Any]:
    """Calls the Modal FastAPI endpoint for topic-based video analysis."""
    if not topic_str:
        return {"status": "error", "error_details": {"message": "Topic cannot be empty."}}
    if not (1 <= max_vids <= 10): # Max 10 as defined in FastAPI endpoint, can adjust
        return {"status": "error", "error_details": {"message": "Max videos must be between 1 and 10."}}

    base_modal_url = os.getenv("MODAL_APP_BASE_URL")
    if not base_modal_url:
        print("ERROR: MODAL_APP_BASE_URL environment variable not set.")
        return {
            "status": "error",
            "error_details": {
                "message": "Modal application base URL is not configured. Please set the MODAL_APP_BASE_URL environment variable."
            }
        }
    topic_endpoint_url = f"{base_modal_url.rstrip('/')}/analyze_topic"

    params = {"topic": topic_str, "max_videos": max_vids}
    print(f"Calling Topic Analysis endpoint: {topic_endpoint_url} with params: {params}")

    try:
        # Using POST as defined in modal_whisper_app.py for /analyze_topic
        response = requests.post(topic_endpoint_url, params=params, timeout=3660) # Long timeout for multiple videos
        response.raise_for_status()
        results = response.json()
        print(f"Received results from Topic Analysis endpoint: {str(results)[:200]}...")
        return results # The endpoint should return the aggregated JSON directly
    except requests.exceptions.Timeout:
        print(f"Request to Topic Analysis endpoint {topic_endpoint_url} timed out.")
        return {"status": "error", "error_details": {"message": "Request to topic analysis service timed out."}}
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error calling Topic Analysis endpoint {topic_endpoint_url}: {e.response.status_code} - {e.response.text}")
        return {"status": "error", "error_details": {"message": f"Topic analysis service returned an error: {e.response.status_code}", "details": e.response.text}}
    except requests.exceptions.RequestException as e:
        print(f"Error calling Topic Analysis endpoint {topic_endpoint_url}: {e}")
        return {"status": "error", "error_details": {"message": "Failed to connect to topic analysis service.", "details": str(e)}}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"status": "error", "error_details": {"message": "An unexpected error occurred during topic analysis call.", "details": str(e)}}

demo_interface = gr.Interface(
    fn=demo_process_video,
    inputs=gr.Textbox(lines=1, label="Video URL or Local File Path", placeholder="Enter YouTube URL, direct video URL, or local file path..."),
    outputs=[gr.Textbox(label="Status"), gr.JSON(label="Comprehensive Analysis Output", scale=2)],
    title="Video Interpretation Demo",
    description="Provide a video URL or local file path to see its transcription status.",
    flagging_options=None
)

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

    with gr.Tab("Interactive Demo"):
        gr.Markdown("### Test the Full Video Analysis Pipeline")
        gr.Markdown("Enter a video URL or local file path to get a comprehensive JSON output including transcription, caption, actions, and objects.")
        with gr.Row():
            text_input = gr.Textbox(lines=1, label="Video URL or Local File Path", placeholder="Enter YouTube URL, direct video URL, or local file path...", scale=3)
        
        analysis_output = gr.JSON(label="Comprehensive Analysis Output", scale=2)
        
        with gr.Row():
            submit_button = gr.Button("Get Comprehensive Analysis", variant="primary", scale=1)
            clear_button = gr.Button("Clear", scale=1)

        # The 'process_video_input' function returns a single dictionary.
        submit_button.click(fn=process_video_input, inputs=[text_input], outputs=[analysis_output])

        def clear_all():
            return [None, None] # Clears text_input and analysis_output
        clear_button.click(fn=clear_all, inputs=[], outputs=[text_input, analysis_output])

        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
                ["http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"],
                # Add a local file path example if you have a common test video, e.g.:
                # ["./sample_video.mp4"] # User would need this file locally
            ],
            inputs=text_input,
            outputs=analysis_output, 
            fn=process_video_input, 
            cache_examples=False, 
        )
        gr.Markdown("**Processing can take several minutes** depending on video length and model inference times. The cache on the Modal backend will speed up repeated requests for the same video.")

    with gr.Tab("Demo (for Manual Testing)"):
        gr.Markdown("### Manually test video URLs or paths for interpretation and observe the JSON response.")
        demo_interface.render()

    with gr.Tab("Topic Video Analysis"):
        gr.Markdown("### Analyze Multiple Videos Based on a Topic")
        gr.Markdown("Enter a topic, and the system will search for relevant videos, analyze them, and provide an aggregated JSON output.")
        
        with gr.Row():
            topic_input = gr.Textbox(label="Enter Topic", placeholder="e.g., 'best cat videos', 'Python programming tutorials'", scale=3)
            max_videos_input = gr.Number(label="Max Videos to Analyze", value=3, minimum=1, maximum=5, step=1, scale=1) # Max 5 for UI, backend might support more
        
        topic_analysis_output = gr.JSON(label="Topic Analysis Results")
        
        with gr.Row():
            topic_submit_button = gr.Button("Analyze Topic Videos", variant="primary")
            topic_clear_button = gr.Button("Clear")

        topic_submit_button.click(
            fn=call_topic_analysis_endpoint, 
            inputs=[topic_input, max_videos_input], 
            outputs=[topic_analysis_output]
        )

        def clear_topic_outputs():
            return [None, 3, None] # topic_input, max_videos_input (reset to default), topic_analysis_output
        topic_clear_button.click(fn=clear_topic_outputs, inputs=[], outputs=[topic_input, max_videos_input, topic_analysis_output])
        
        gr.Examples(
            examples=[
                ["AI in healthcare", 2],
                ["sustainable energy solutions", 3],
                ["how to make sourdough bread", 1]
            ],
            inputs=[topic_input, max_videos_input],
            outputs=topic_analysis_output,
            fn=call_topic_analysis_endpoint,
            cache_examples=False
        )
        gr.Markdown("**Note:** This process involves searching for videos and then analyzing each one. It can take a significant amount of time, especially for multiple videos. The backend has a long timeout, but please be patient.")

# Launch the Gradio application
if __name__ == "__main__":
    app.launch(debug=True, server_name="0.0.0.0")
