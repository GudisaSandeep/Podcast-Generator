# AI Podcast Generator

A powerful and versatile podcast generation tool that converts various types of content into engaging audio conversations using Gemini AI and Text-to-Speech technology.

## Features

- **Topic-Based Podcasts**: Generate podcasts from any topic or subject matter
- **PDF Document Conversion**: Turn PDF documents into conversational podcasts
- **YouTube Content**: Convert YouTube videos into podcast discussions using transcripts
- **Web Article Conversion**: Transform any web article into an engaging podcast
- **Multi-Language Support**: Generate podcasts in multiple languages
- **Customizable Voices**: Choose from 30 different AI voices for podcast hosts
- **Adjustable Duration**: Select short, medium, or long podcast formats
- **Professional Conversations**: AI-generated natural-sounding dialogues
- **Modern Web Interface**: Easy-to-use Gradio-based UI

## Prerequisites

- Python 3.8 or higher
- Google Cloud account with Gemini API access
- Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd podcast-generator
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python podcast_gradio.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

### Generating Topic-Based Podcasts
1. Select the "Topic-Based Podcast" tab
2. Enter your desired topic
3. Choose the language
4. Select podcast duration (short/medium/long)
5. Customize host names and voices
6. Click "Generate Podcast from Topic"

### Converting PDF Documents
1. Select the "PDF-Based Podcast" tab
2. Upload your PDF file
3. Configure language and duration preferences
4. Customize host settings
5. Click "Generate Podcast from PDF"

### Creating Podcasts from YouTube Videos
1. Select the "YouTube-Based Podcast" tab
2. Enter a valid YouTube URL
3. Set your preferred language and duration
4. Customize host voices
5. Click "Generate Podcast from YouTube"

### Converting Web Articles
1. Select the "Web Article Podcast" tab
2. Enter the article URL
3. Choose language and duration settings
4. Configure host preferences
5. Click "Generate Podcast from Web Article"

## Available Voices

The application supports the following AI voices:
- achernar
- achird
- algenib
- algieba
- alnilam
- aoede
- autonoe
- callirrhoe
- charon
- despina
- enceladus
- erinome
- fenrir
- gacrux
- iapetus
- kore
- laomedeia
- leda
- orus
- puck
- pulcherrima
- rasalgethi
- sadachbia
- sadaltager
- schedar
- sulafat
- umbriel
- vindemiatrix
- zephyr
- zubenelgenubi

## Limitations

- YouTube conversion requires available transcripts
- Web article conversion depends on website accessibility and structure
- Maximum content length limitations apply for each source type
- Some websites may block web scraping

## Troubleshooting

1. **API Key Issues**:
   - Ensure your Gemini API key is correctly set in the `.env` file
   - Verify your API key has the necessary permissions

2. **YouTube Processing Errors**:
   - Check if the video has available transcripts
   - Verify the URL is correct and the video is accessible

3. **PDF Processing Issues**:
   - Ensure the PDF is text-based and not scanned images
   - Check if the file size is within reasonable limits

4. **Web Scraping Errors**:
   - Verify the website allows content scraping
   - Check if the URL is accessible from your location

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Acknowledgments

- Google Gemini AI for text generation and TTS
- Gradio for the web interface
- BeautifulSoup4 for web scraping
- YouTube Transcript API for video processing 