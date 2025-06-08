---
title: Contextual Video Data Server (MCP Tool/Server) - The Ultimate Video Whisperer!
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: "MCP server extracting transcriptions, captions, and actions from videos for LLMs. Agents-MCP-Hackathon entry."
tags: ["mcp-server-track"]
---

# 🎬 Contextual Video Data Server (MCP Tool/Server) - The Ultimate Video Whisperer! 🚀

Welcome to the **Contextual Video Data MCP Server**! This isn't just any project; it's our glorious entry into Track 1 ("MCP Tool / Server") of the **Agents-MCP-Hackathon**! 🏆 Our mission, should we choose to accept it (and we totally do!), is to build an MCP (Model Context Protocol) server that acts like a super-smart assistant for Large Language Models (LLMs), feeding them rich, juicy, contextual information extracted directly from videos. Think of it as giving LLMs eyes *and* ears for the video world! 👀👂

Our grand vision? To take any video you throw at us (URLs, direct uploads, maybe even a carrier pigeon with a USB stick 🕊️... okay, maybe not the pigeon) and dissect it. We're talking: 
1.  Flawless **audio transcriptions** (what's being said).
2.  And the really exciting part: comprehensive **visual interpretations** (what's being seen and done)! This includes:
    *   **Video Captioning**: A snappy summary of the video's content.
    *   **Action Recognition**: Identifying all the cool (or mundane) actions happening.
    *   **Object Detection/Tracking**: Pinpointing and following key objects through the frames.

All this delicious data will be neatly packaged and served up via an MCP-compliant API endpoint, ready for LLMs to gobble up and become even more insightful. Let's make those AI brains BIGGER! 🧠💥

## 🏅 Hackathon Context (Track 1: MCP Tool / Server) - Our Quest!

This project is laser-focused on conquering Track 1 of the Agents-MCP-Hackathon. Here's how we're hitting all the right notes 🎶:

-   **MCP Server/Tool Extraordinaire**: We're building a Gradio application, destined to live on Hugging Face Spaces, that proudly stands as an MCP server. It's the digital butler for video context!
-   **Supercharging LLMs**: Our server's raison d'être is to provide video-derived context, empowering LLMs to deliver responses that are not just smart, but *video-smart*.
-   **Showtime! The Demo Requirement**: We know talk is cheap. That's why a crucial part of our submission will be a dazzling video demonstration (linked right here in this README, eventually!). It'll showcase our MCP server strutting its stuff with an external MCP Client (like Claude Desktop, Cursor, Tiny Agents, or even another cool Gradio app). You'll see the magic unfold: video in ➡️ context out ➡️ happy LLM. 🪄

## 🏗️ Project Architecture - The Three Musketeers of Video Context!

Our system isn't just thrown together; it's a masterpiece of engineering (if we do say so ourselves 😉). We've got a refined three-tier architecture, ensuring everyone plays their part perfectly:

1.  **Gradio App (This Hugging Face Space - Our MCP Server HQ 🏰)**: 
    *   The welcoming face of our operation! Handles video uploads (URLs or your precious files).
    *   The grand conductor 🎻, orchestrating the video processing symphony by calling in our Modal backend.
    *   The meticulous librarian 📚, structuring all the extracted goodies (transcriptions, and soon, a smorgasbord of visual data).
    *   The ever-ready API provider, serving up context via an MCP-compliant endpoint (think `gr.JSON()` or a slick FastAPI route) to any LLM frontend that comes knocking.
    *   **Super Important Note**: This Gradio app is a dedicated MCP server. No direct LLM chit-chat here! It's all about processing videos and serving data, keeping things clean and focused. ✨

2.  **Modal Backend (The Heavy Lifter 💪 - Our Digital Hercules!)**:
    *   This is where the real grunt work happens. Got a computationally intensive task? Modal's on it! 
    *   Currently wrestling with audio extraction and Whisper transcriptions using behemoth models like `openai/whisper-large-v3`.
    *   Gearing up to tackle our **Triple Threat Video Analysis™**: captioning, action recognition, and object detection/tracking. It's gonna be epic!
    *   Summoned by the Gradio App, it delivers efficiency and scalability like a champ. 🥊

3.  **Another Hugging Face Space (The LLM's Frontend Friend 🤖 - *Not Our Circus, Not Our Monkeys for this Task*)**:
    *   Imagine this as an external buddy project – the cool app where end-users chat with an LLM (Claude, Llama, you name it).
    *   This buddy will call *our* Gradio MCP Server to get the video lowdown.
    *   Then, armed with our context, it'll help the LLM craft super-duper responses.
    *   This separation of powers is key! It keeps our Contextual Video Data Server lean, mean, and a top-notch MCP server, just what the hackathon ordered.

## ✨ Features - What Makes Us Sparkle!

-   📥 **Versatile Video Ingestion**: Handles YouTube links (thanks, `yt-dlp`!) and direct file uploads with grace.
-   🎤 **Crystal-Clear Transcriptions**: Leverages the mighty Whisper models on Modal for top-tier audio-to-text conversion.
-   🤝 **MCP-Compliant API**: Serves up structured JSON data (transcriptions now, a feast of video analysis soon!) via a well-defined API endpoint.
-   🖥️ **User-Friendly Gradio UI**: A simple, intuitive interface for uploading videos and seeing the magic happen (locally for now, soon on HF Spaces!).
-   🌟 **THE BIG ONE (PLANNED!): Parallel Multi-Modal Video Interpretation!** 🌟
    *   🖼️ **Video Captioning**: What's the gist of this video?
    *   🏃 **Action Recognition**: Who's doing what? (Running? Jumping? Contemplating the universe?)
    *   🔍 **Object Detection/Tracking**: What's in the scene, and where is it going? (Is that a cat or a very fluffy loaf of bread? 🍞)
    *   All processed in parallel (because we're ambitious like that!) and presented in an LLM-friendly format.

## 🗺️ Our Epic Development Journey & Discoveries So Far! (The Chronicles of Context 📜)

This hasn't been just coding; it's been an adventure! Here are some highlights from our quest for video understanding:

-   **The Foundation**: We started by bravely integrating `yt-dlp` (for taming wild YouTube videos) and `moviepy` (our trusty audio-extracting squire).
-   **The Heart of Transcription**: We summoned the power of Whisper on Modal, using the Hugging Face `transformers` pipeline as our spellbook. ếm
-   **The Ascent of Models (A Tale of Quality)**: 
    -   Our quest began humbly with `openai/whisper-base`.
    -   We then climbed the ladder to `openai/whisper-small`, `openai/whisper-medium`, and have now reached the peak (for now!) with `openai/whisper-large-v3`, all in pursuit of transcription perfection! 🏔️
-   **Taming the Parameters**: We dueled with `temperature` and `no_repeat_ngram_size` (our secret weapons in `generate_kwargs`) to banish repetitive demons and ensure coherent narratives.
-   **Speaking the Right Language**: We wisely added `language="en"` to `generate_kwargs`, ensuring Whisper knew what to expect (no more accidental Klingon transcriptions!).
-   **Architectural Epiphanies**: Like master builders, we refined our design to the glorious three-tier architecture you see today – a beacon of clarity and scalability, fit for MCP royalty. 👑
-   **Slaying Dragons (aka Debugging)**: We've vanquished pesky bugs, from tricky parameter passing to the Hugging Face pipeline to ensuring our Modal environment's dependencies were as harmonious as a barbershop quartet.  barbershop

## 📁 Project Structure - Know Your Way Around!

-   `app.py`: The command center! Our main Gradio application (MCP Server), handling user interactions and the all-important API endpoint.
-   `modal_whisper_app.py`: The engine room! Defines the Modal app and the functions that do the heavy lifting (`transcribe_video_audio` and its future video-analyzing siblings).
-   `requirements.txt`: The shopping list for our local `app.py`'s Python dependencies.
-   `README.md`: You're looking at it! Our project's story, map, and instruction manual, all rolled into one. 🗺️

## 🛠️ Setup - Let's Get This Party Started!

### Prerequisites - The Essentials Before the Magic!

-   Python 3.10+ (because we like our Python fresh! 🐍)
-   A Modal account, with the CLI installed and configured (`pip install modal-client`, then `modal setup`). This is your key to the Modal kingdom! 🔑
-   `ffmpeg` installed locally. This digital Swiss Army knife is crucial for `yt-dlp` and `moviepy`.
    -   Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    -   macOS (Homebrew): `brew install ffmpeg`

### Local Setup - Your Very Own Context Server!

1.  **Clone the Treasure Chest (Our Repository)!**
    ```bash
    # Make sure this URL points to our magnificent repo!
    git clone https://github.com/jomasego/video_mcp.git 
    cd video_mcp
    ```

2.  **Install the Local Spells (Dependencies)!**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Unleash the Modal Beast (Deploy the Function)!**
    Make sure your Modal CLI is logged in and ready to rumble!
    ```bash
    modal deploy modal_whisper_app.py
    ```

### Running the Local Application (Our MCP Server in Action!)

1.  **Ignite the Gradio App!**
    ```bash
    python3 app.py
    ```
2.  Point your trusty web browser to the URL Gradio provides (usually a friendly `http://127.0.0.1:7860`).
3.  Feed it videos! Watch it work! Marvel at its (soon-to-be-even-more-awesome) power! 🤩

## 🔮 Modal Function Details - The Wizardry Behind the Curtain!

The `modal_whisper_app.py` script is where the real enchantment happens. It defines Modal functions that:
-   Live in a custom Docker image, armed with `ffmpeg`, `transformers`, `torch`, `moviepy`, `soundfile`, and `huggingface_hub` – all the tools a growing video AI needs!
-   Currently, `transcribe_video_audio` bravely takes video bytes, extracts the audio, and transcribes it using our chosen Whisper champion (defaulting to the mighty `openai/whisper-large-v3`).
-   **Coming Soon**: New functions (or an upgraded mega-function!) to perform the Triple Threat Video Analysis™ (captioning, action recognition, object detection).
-   Might need your Hugging Face token (via a Modal secret like `HF_TOKEN_SECRET`) if we dabble in gated models or just want to be polite to the Hugging Face Hub. 🤗

## 📡 API Endpoint & MCP Integration Plan - Talking to the LLMs!

Our Gradio app (`app.py`) isn't just a pretty face; it's a communication hub! It'll expose a robust API endpoint for MCP clients.

-   **How We'll Build It**: We're thinking a Gradio Interface function with a catchy `api_name`, or maybe a sleek FastAPI route snuggled into our Gradio app. Options, options! 🤓
-   **The Language of LLMs (Output Format)**: Our API will speak fluent JSON. It'll provide a beautifully structured package containing the transcription, and soon, all the video analysis gold (captions, actions, objects). Imagine something like:
    ```json
    {
      "transcription": "The quick brown fox jumps over the lazy dog...",
      "video_caption": "A montage of adorable animal antics.",
      "actions_detected": ["jumping", "sleeping", "being generally cute"],
      "objects_of_interest": [
        {"label": "fox", "confidence": 0.95, "bounding_box": [10, 20, 50, 60]},
        {"label": "dog", "confidence": 0.98, "bounding_box": [100, 120, 80, 70]}
      ]
    }
    ```
    (Okay, the exact structure is TBD, but you get the idea – rich and LLM-ready!)
-   **The Grand Demo Video 🎬**: As per the sacred hackathon scrolls, we *will* create a video. This video will be our magnum opus, showcasing an MCP client (Claude Desktop? Cursor? A plucky custom Gradio client?) calling our API, fetching this glorious context, and using it to achieve new heights of LLM wisdom. We're still picking our co-star for this demo, so stay tuned! 🌟

## 🚀 Future Work & Next Steps - To Infinity and Beyond!

We're not stopping at 'good enough'; we're aiming for 'mind-blowingly awesome'! 🤯

-   **Unleash the Triple Threat Video Analysis™!**
    -   Scour the Hugging Face Hub and beyond for the best models for video captioning, action recognition, and object detection/tracking. 🕵️‍♂️
    -   Integrate these champions into new or existing Modal functions. We want parallel processing for maximum speed and insight!
    -   Upgrade our Gradio app and API to proudly present these new layers of video understanding.
-   **Fortify the API Endpoint**: Make it rock-solid. Bulletproof error handling. Consistent, crystal-clear output. An API so good, MCP clients will write songs about it. 🎶
-   **Choose Our Champion (MCP Client) & Film the Epic Demo**: Finalize which MCP client will star alongside our server in the demo video. Then, lights, camera, action! 🎥 We need a compelling showcase for the hackathon judges.
-   **The Never-Ending Quest for Perfection**: Continuously refine transcription accuracy. Squeeze out every last drop of processing speed. Optimize costs. The journey never truly ends! 🛤️

## 🤯 Troubleshooting - Don't Panic! (Usually...)

When the digital gremlins strike, here are a few things to check:

-   **`ModuleNotFoundError: No module named 'moviepy.editor'` (lurking in Modal logs):**
    Ah, the classic! `moviepy` might be playing hide-and-seek in your Modal image. Double-check `pip_install` and any `run_commands` in `modal_whisper_app.py`. A redeploy might be in order.
-   **`yt-dlp` throwing a tantrum or `ffmpeg` acting shy:**
    Ensure `ffmpeg` is installed both locally (for `app.py`'s antics) AND within the Modal image (`apt_install("ffmpeg")`). It needs to be everywhere, like a helpful ninja. 🥷
-   **Modal Authentication Woes or Deployment Drama:**
    Did you run `modal setup`? Is your Modal token still feeling loved and active? If you're deploying to Hugging Face Spaces, remember Modal tokens might need special treatment as environment variables/secrets. Check the scrolls (aka Modal docs)! 📜

--- 
*Let's build something amazing!* 🌟💻🎉

