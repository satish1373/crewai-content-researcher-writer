# CrewAI content-researcher-writer Gradio App 🚀

This project hosts a Gradio web app that uses CrewAI and CrewAI Tools to research and generate content on a given topic using Gemini LLM.

## Features
- Web-based Gradio interface
- Multi-agent research + content generation
- Uses Gemini API securely (via environment variables)

## How to run locally
```bash
git clone https://github.com/your-username/crewai-gradio-app.git
cd crewai-gradio-app
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py

