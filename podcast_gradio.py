import gradio as gr
from google import genai
from google.genai import types
import wave
from dotenv import load_dotenv
import os
import tempfile
import base64
import PyPDF2
import io
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import subprocess
import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import html2text

# Load environment variables
load_dotenv()

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save audio data as a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return filename

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    if pdf_file is None:
        return ""
    
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {str(e)}"
    
    return text

def get_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    # For URLs like: https://www.youtube.com/watch?v=VIDEO_ID
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                return query_params['v'][0]
    
    # For URLs like: https://youtu.be/VIDEO_ID
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    
    # For embed URLs: https://www.youtube.com/embed/VIDEO_ID
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com') and parsed_url.path.startswith('/embed/'):
        return parsed_url.path.split('/embed/')[1]
    
    return None

def get_video_info(video_id):
    """Get video title and info using YouTube Data API or fallback to oembed"""
    try:
        # Use YouTube oEmbed API (doesn't require API key)
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get('title', 'Unknown Title'),
                "author": data.get('author_name', 'Unknown Author'),
                "success": True
            }
    except Exception as e:
        pass
    
    # Fallback to basic info
    return {
        "title": f"YouTube Video (ID: {video_id})",
        "author": "Unknown Creator",
        "success": True
    }

def get_youtube_transcript(youtube_url):
    """Get transcript from YouTube video"""
    try:
        # Extract video ID from URL
        video_id = get_video_id(youtube_url)
        if not video_id:
            return {
                "success": False,
                "error": "Could not extract video ID from URL. Please check the URL format."
            }
        
        # Get video info
        video_info = get_video_info(video_id)
        
        # Get available transcripts
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript first
            transcript_text = ""
            try:
                transcript = transcript_list.find_transcript(['en'])
                transcript_parts = transcript.fetch()
                for part in transcript_parts:
                    transcript_text += part['text'] + " "
            except NoTranscriptFound:
                # If no English transcript, use the first available one
                transcript = transcript_list.find_transcript(['en-US', 'en-GB', 'auto'])
                transcript_parts = transcript.fetch()
                for part in transcript_parts:
                    transcript_text += part['text'] + " "
            
            return {
                "transcript": transcript_text,
                "title": video_info.get("title", "YouTube Video"),
                "author": video_info.get("author", "Unknown"),
                "success": True
            }
            
        except TranscriptsDisabled:
            return {
                "success": False,
                "error": "Transcripts are disabled for this video. Please try a different video."
            }
        except NoTranscriptFound:
            return {
                "success": False,
                "error": "No transcript available for this video. Please try a different video."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving transcript: {str(e)}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to process video transcript: {str(e)}"
        }

def transcribe_youtube_audio(audio_path):
    """Transcribe YouTube audio using Gemini"""
    try:
        # Convert audio to wav format if needed
        temp_dir = os.path.dirname(audio_path)
        wav_path = os.path.join(temp_dir, "audio.wav")
        
        # Use ffmpeg to convert to wav
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            wav_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Read the wav file as binary data
        with open(wav_path, "rb") as f:
            audio_data = f.read()
        
        # Encode to base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        
        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create prompt for transcription
        prompt = "Please transcribe this audio accurately. Include speaker changes if detected."
        
        # Send to Gemini for transcription
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}
                    ]
                }
            ]
        )
        
        return {
            "transcript": response.text,
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_podcast_from_youtube(youtube_url, language, host1_name, host2_name, host1_voice, host2_voice, duration="medium"):
    """Generate a podcast from YouTube video transcript"""
    # Basic URL validation
    if not youtube_url or not youtube_url.strip():
        return None, "Please enter a valid YouTube URL"
        
    if not is_valid_youtube_url(youtube_url):
        return None, "The URL does not appear to be a valid YouTube video URL"
    
    # Get video transcript
    transcript_result = get_youtube_transcript(youtube_url)
    if not transcript_result["success"]:
        return None, f"Failed to get YouTube transcript: {transcript_result['error']}"
    
    # Get video metadata
    video_title = transcript_result["title"]
    video_author = transcript_result["author"]
    transcript = transcript_result["transcript"]
    
    # Summarize if needed (for longer transcripts)
    if len(transcript) > 3000:
        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        summary_prompt = f"""Summarize the following transcript from a YouTube video titled "{video_title}" by {video_author}:
        
        {transcript[:30000]}  # Limit to 30k chars if extremely long
        
        Provide a concise summary capturing the main points and key insights.
        """
        
        summary_response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=summary_prompt
        )
        content_to_discuss = summary_response.text
    else:
        content_to_discuss = transcript
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Determine duration instruction
    duration_text = ""
    if duration == "short":
        duration_text = "for about 30 seconds"
    elif duration == "medium":
        duration_text = "for about 1-2 minutes"
    else:
        duration_text = "for about 3-5 minutes"
    
    # Generate podcast transcript
    prompt = f"""Create a conversational podcast script between {host1_name} and {host2_name} discussing this YouTube video titled "{video_title}" by {video_author} {duration_text}.
              The conversation should be in {language} language.
              Make it engaging, informative, and natural-sounding with back-and-forth dialogue.
              Hosts should discuss key points, share insights, and explain concepts in an accessible way.
              
              VIDEO CONTENT TO DISCUSS:
              {content_to_discuss}
              
              Begin with a brief introduction to the video topic, then explore the main ideas.
              """
    
    podcast_transcript = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    ).text
    
    # Generate audio from transcript
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=podcast_transcript,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=host1_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host1_voice,
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker=host2_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host2_voice,
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )
    
    # Get audio data and save to temporary file
    data = response.candidates[0].content.parts[0].inline_data.data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = wave_file(temp_file.name, data)
    
    return audio_path, podcast_transcript

def summarize_pdf_content(pdf_text, max_length=1000):
    """Summarize the content of a PDF using Gemini"""
    if not pdf_text:
        return ""
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Truncate if too long
    if len(pdf_text) > 30000:
        pdf_text = pdf_text[:30000]
    
    prompt = f"""Summarize the following document content for a podcast discussion.
    Extract the key points, main arguments, and interesting facts.
    Keep the summary concise (around 300-500 words).
    
    DOCUMENT CONTENT:
    {pdf_text}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        summary = response.text
        return summary
    except Exception as e:
        return f"Error summarizing PDF: {str(e)}"

def generate_podcast_from_pdf(pdf_file, language, host1_name, host2_name, host1_voice, host2_voice, duration="medium"):
    """Generate a podcast conversation from a PDF file"""
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_file)
    if not pdf_text or pdf_text.startswith("Error"):
        return None, f"Failed to process PDF: {pdf_text}"
    
    # Summarize the PDF content
    summary = summarize_pdf_content(pdf_text)
    if not summary or summary.startswith("Error"):
        return None, f"Failed to summarize PDF: {summary}"
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Determine duration instruction
    duration_text = ""
    if duration == "short":
        duration_text = "for about 30 seconds"
    elif duration == "medium":
        duration_text = "for about 1-2 minutes"
    else:
        duration_text = "for about 3-5 minutes"
    
    # Generate transcript
    prompt = f"""Create a conversational podcast script between {host1_name} and {host2_name} discussing the following content {duration_text}.
              The conversation should be in {language} language.
              Make it engaging, informative, and natural-sounding with back-and-forth dialogue.
              Hosts should discuss key points, share insights, and explain concepts in an accessible way.
              
              CONTENT TO DISCUSS:
              {summary}
              """
    
    transcript = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    ).text
    
    # Generate audio from transcript
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=transcript,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=host1_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host1_voice,
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker=host2_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host2_voice,
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )
    
    # Get audio data and save to temporary file
    data = response.candidates[0].content.parts[0].inline_data.data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = wave_file(temp_file.name, data)
    
    return audio_path, transcript

def generate_podcast(topic, language, host1_name, host2_name, host1_voice, host2_voice, duration="short"):
    """Generate a podcast conversation and return the audio file"""
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Determine duration instruction
    duration_text = ""
    if duration == "short":
        duration_text = "for about 10-15 seconds"
    elif duration == "medium":
        duration_text = "for about 30-45 seconds"
    else:
        duration_text = "for about 1-2 minutes"
    
    # Generate transcript
    prompt = f"""Generate a conversation transcript about {topic} in {language} language {duration_text}.
              The hosts names are {host1_name} and {host2_name}.
              Make it engaging and informative with back-and-forth dialogue between hosts."""
    
    transcript = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    ).text
    
    # Generate audio from transcript
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=transcript,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=host1_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host1_voice,
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker=host2_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host2_voice,
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )
    
    # Get audio data and save to temporary file
    data = response.candidates[0].content.parts[0].inline_data.data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = wave_file(temp_file.name, data)
    
    return audio_path, transcript

# Function to validate YouTube URL
def is_valid_youtube_url(url):
    if not url or not url.strip():
        return False
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(youtube_regex, url)
    return match is not None

# List of available voices in Gemini
voice_options = [
    "achernar", "achird", "algenib", "algieba", "alnilam", "aoede", 
    "autonoe", "callirrhoe", "charon", "despina", "enceladus", 
    "erinome", "fenrir", "gacrux", "iapetus", "kore", "laomedeia", 
    "leda", "orus", "puck", "pulcherrima", "rasalgethi", "sadachbia", 
    "sadaltager", "schedar", "sulafat", "umbriel", "vindemiatrix", 
    "zephyr", "zubenelgenubi"
]

def scrape_web_content(url):
    """Scrape content from a web URL"""
    try:
        # Send request with headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "Web Article"
        
        # Get main content
        # First try to find article or main content
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=['content', 'article', 'post'])
        
        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to all paragraph text
            content = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
        
        # Convert HTML to plain text
        h = html2text.HTML2Text()
        h.ignore_links = True
        content = h.handle(content)
        
        return {
            "success": True,
            "title": title,
            "content": content,
            "url": url
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to scrape content: {str(e)}"
        }

def generate_podcast_from_url(url, language, host1_name, host2_name, host1_voice, host2_voice, duration="medium"):
    """Generate a podcast from web content"""
    # Scrape web content
    content_result = scrape_web_content(url)
    if not content_result["success"]:
        return None, f"Failed to process URL: {content_result['error']}"
    
    # Get content and title
    content = content_result["content"]
    title = content_result["title"]
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Summarize if content is too long
    if len(content) > 3000:
        summary_prompt = f"""Summarize the following web article titled "{title}":
        
        {content[:30000]}  # Limit to 30k chars if extremely long
        
        Provide a concise summary capturing the main points, key insights, and important details.
        """
        
        summary_response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=summary_prompt
        )
        content_to_discuss = summary_response.text
    else:
        content_to_discuss = content
    
    # Determine duration instruction
    duration_text = ""
    if duration == "short":
        duration_text = "for about 30 seconds"
    elif duration == "medium":
        duration_text = "for about 1-2 minutes"
    else:
        duration_text = "for about 3-5 minutes"
    
    # Generate podcast transcript
    prompt = f"""Create a conversational podcast script between {host1_name} and {host2_name} discussing this web article titled "{title}" {duration_text}.
              The conversation should be in {language} language.
              Make it engaging, informative, and natural-sounding with back-and-forth dialogue.
              Hosts should discuss key points, share insights, and explain concepts in an accessible way.
              
              ARTICLE CONTENT TO DISCUSS:
              {content_to_discuss}
              
              Begin with a brief introduction to the article topic, then explore the main ideas.
              """
    
    podcast_transcript = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    ).text
    
    # Generate audio from transcript
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=podcast_transcript,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=host1_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host1_voice,
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker=host2_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=host2_voice,
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )
    
    # Get audio data and save to temporary file
    data = response.candidates[0].content.parts[0].inline_data.data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = wave_file(temp_file.name, data)
    
    return audio_path, podcast_transcript

def is_valid_url(url):
    """Validate if a string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Create Gradio interface
with gr.Blocks(title="AI Podcast Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Podcast Generator")
    gr.Markdown("Generate conversations between two hosts from topics, PDF documents, YouTube videos, or web articles.")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Topic-Based Podcast"):
            with gr.Row():
                with gr.Column():
                    topic_input = gr.Textbox(
                        label="Topic", 
                        placeholder="Enter the topic for the podcast (e.g., 'DeepFake Scams', 'Climate Change')",
                        value="DeepFake Scams"
                    )
                    
                    language_input_topic = gr.Textbox(
                        label="Language", 
                        placeholder="Enter the language (e.g., 'English', 'Telugu', 'Spanish')",
                        value="English"
                    )
                    
                    duration_input_topic = gr.Radio(
                        choices=["short", "medium", "long"],
                        label="Duration",
                        value="short"
                    )
                    
                with gr.Column():
                    host1_name_topic = gr.Textbox(label="Host 1 Name", value="Kamala")
                    host1_voice_topic = gr.Dropdown(choices=voice_options, label="Host 1 Voice", value="achernar")
                    
                    host2_name_topic = gr.Textbox(label="Host 2 Name", value="Subbarao")
                    host2_voice_topic = gr.Dropdown(choices=voice_options, label="Host 2 Voice", value="achird")
                    
            generate_topic_btn = gr.Button("Generate Podcast from Topic", variant="primary")
            
        with gr.TabItem("PDF-Based Podcast"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="binary"
                    )
                    
                    language_input_pdf = gr.Textbox(
                        label="Language", 
                        placeholder="Enter the language (e.g., 'English', 'Telugu', 'Spanish')",
                        value="English"
                    )
                    
                    duration_input_pdf = gr.Radio(
                        choices=["short", "medium", "long"],
                        label="Duration",
                        value="medium",
                        info="Short: ~30s, Medium: ~1-2m, Long: ~3-5m"
                    )
                    
                with gr.Column():
                    host1_name_pdf = gr.Textbox(label="Host 1 Name", value="Kamala")
                    host1_voice_pdf = gr.Dropdown(choices=voice_options, label="Host 1 Voice", value="achernar")
                    
                    host2_name_pdf = gr.Textbox(label="Host 2 Name", value="Subbarao")
                    host2_voice_pdf = gr.Dropdown(choices=voice_options, label="Host 2 Voice", value="achird")
                    
            generate_pdf_btn = gr.Button("Generate Podcast from PDF", variant="primary")
            
        with gr.TabItem("YouTube-Based Podcast"):
            with gr.Row():
                with gr.Column():
                    youtube_url_input = gr.Textbox(
                        label="YouTube URL", 
                        placeholder="Enter YouTube video URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)",
                    )
                    
                    language_input_youtube = gr.Textbox(
                        label="Language", 
                        placeholder="Enter the language (e.g., 'English', 'Telugu', 'Spanish')",
                        value="English"
                    )
                    
                    duration_input_youtube = gr.Radio(
                        choices=["short", "medium", "long"],
                        label="Duration",
                        value="medium",
                        info="Short: ~30s, Medium: ~1-2m, Long: ~3-5m"
                    )
                    
                with gr.Column():
                    host1_name_youtube = gr.Textbox(label="Host 1 Name", value="Kamala")
                    host1_voice_youtube = gr.Dropdown(choices=voice_options, label="Host 1 Voice", value="achernar")
                    
                    host2_name_youtube = gr.Textbox(label="Host 2 Name", value="Subbarao")
                    host2_voice_youtube = gr.Dropdown(choices=voice_options, label="Host 2 Voice", value="achird")
                    
            generate_youtube_btn = gr.Button("Generate Podcast from YouTube", variant="primary")
        
        with gr.TabItem("Web Article Podcast"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        label="Web Article URL", 
                        placeholder="Enter the URL of any web article (e.g., https://example.com/article)",
                    )
                    
                    language_input_url = gr.Textbox(
                        label="Language", 
                        placeholder="Enter the language (e.g., 'English', 'Telugu', 'Spanish')",
                        value="English"
                    )
                    
                    duration_input_url = gr.Radio(
                        choices=["short", "medium", "long"],
                        label="Duration",
                        value="medium",
                        info="Short: ~30s, Medium: ~1-2m, Long: ~3-5m"
                    )
                    
                with gr.Column():
                    host1_name_url = gr.Textbox(label="Host 1 Name", value="Kamala")
                    host1_voice_url = gr.Dropdown(choices=voice_options, label="Host 1 Voice", value="achernar")
                    
                    host2_name_url = gr.Textbox(label="Host 2 Name", value="Subbarao")
                    host2_voice_url = gr.Dropdown(choices=voice_options, label="Host 2 Voice", value="achird")
                    
            generate_url_btn = gr.Button("Generate Podcast from Web Article", variant="primary")
    
    with gr.Row():
        with gr.Column():
            audio_output = gr.Audio(label="Generated Podcast", type="filepath")
        with gr.Column():
            transcript_output = gr.Textbox(label="Generated Transcript", lines=10)
    
    # Add validation for YouTube URL - modified to not clear the input
    youtube_url_input.change(
        fn=lambda url: gr.update(value=url),  # Just keep the URL as is
        inputs=youtube_url_input,
        outputs=youtube_url_input
    )
    
    # Connect buttons to functions
    generate_topic_btn.click(
        fn=generate_podcast,
        inputs=[
            topic_input, 
            language_input_topic, 
            host1_name_topic, 
            host2_name_topic, 
            host1_voice_topic, 
            host2_voice_topic, 
            duration_input_topic
        ],
        outputs=[audio_output, transcript_output]
    )
    
    generate_pdf_btn.click(
        fn=generate_podcast_from_pdf,
        inputs=[
            pdf_input, 
            language_input_pdf, 
            host1_name_pdf, 
            host2_name_pdf, 
            host1_voice_pdf, 
            host2_voice_pdf, 
            duration_input_pdf
        ],
        outputs=[audio_output, transcript_output]
    )
    
    generate_youtube_btn.click(
        fn=generate_podcast_from_youtube,
        inputs=[
            youtube_url_input,
            language_input_youtube,
            host1_name_youtube,
            host2_name_youtube,
            host1_voice_youtube,
            host2_voice_youtube,
            duration_input_youtube
        ],
        outputs=[audio_output, transcript_output]
    )
    
    # Add validation for URL
    url_input.change(
        fn=lambda url: gr.update(value=url),
        inputs=url_input,
        outputs=url_input
    )
    
    # Connect URL button to function
    generate_url_btn.click(
        fn=generate_podcast_from_url,
        inputs=[
            url_input,
            language_input_url,
            host1_name_url,
            host2_name_url,
            host1_voice_url,
            host2_voice_url,
            duration_input_url
        ],
        outputs=[audio_output, transcript_output]
    )
    
    gr.Markdown("## How to Use")
    gr.Markdown("""
    ### Topic-Based Podcasts
    1. Enter a topic for your podcast
    2. Choose the language
    3. Set the duration
    4. Customize host names and voices
    5. Click 'Generate Podcast from Topic'
    
    ### PDF-Based Podcasts
    1. Upload a PDF document
    2. Choose the language for the podcast
    3. Set the duration
    4. Customize host names and voices
    5. Click 'Generate Podcast from PDF'
    
    ### YouTube-Based Podcasts
    1. Enter a valid YouTube video URL
    2. Choose the language for the podcast
    3. Set the duration
    4. Customize host names and voices
    5. Click 'Generate Podcast from YouTube'
    
    ### Web Article Podcasts
    1. Enter any web article URL
    2. Choose the language for the podcast
    3. Set the duration
    4. Customize host names and voices
    5. Click 'Generate Podcast from Web Article'
    """)
    
    gr.Markdown("### Note")
    gr.Markdown("This application requires a valid Gemini API key in your .env file. YouTube processing uses transcripts when available. Web scraping capabilities may vary depending on the website's structure and access restrictions.")

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch() 