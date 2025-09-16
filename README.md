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

CrewAI Content-Researcher-Writer
This is a powerful, AI-driven application designed to automate the process of researching and writing content. By leveraging the CrewAI framework, this project orchestrates a team of specialized AI agents to generate high-quality, well-researched, and structured articles or reports on a given topic.

🧠 How it Works
The application operates on a multi-agent system, where each agent has a specific role and set of tools:

Researcher Agent: This agent's primary task is to find and gather information on the specified topic. It is equipped with tools to perform web searches, analyze online sources, and collect relevant data.

Writer Agent: This agent receives the research findings from the Researcher. Its role is to synthesize the information, structure it logically, and write a comprehensive and engaging article.

Editor Agent: The Editor reviews the drafted content for grammar, clarity, style, and accuracy. It ensures the final output is polished and meets a high standard of quality.

✨ Features
Automated Content Generation: Go from a topic idea to a complete article with minimal manual input.

Customizable Workflow: The roles and tasks of each agent can be easily modified to suit your specific content needs.

Structured Output: The final output is a well-organized article, ready for publication.

Extensible Tools: Agents can be equipped with new tools (e.g., for accessing databases, analyzing sentiment, or creating visualizations) to enhance their capabilities.

🚀 Getting Started
Prerequisites
Python 3.8+

pip package manager

A valid API key for a large language model (e.g., OpenAI, Anthropic).

Installation
Clone this repository:

git clone [https://github.com/your-username/crewai-content-researcher-writer.git](https://github.com/your-username/crewai-content-researcher-writer.git)
cd crewai-content-researcher-writer

Install the required Python packages:

pip install -r requirements.txt

Set up your API key by creating a .env file in the project root with the following content:

OPENAI_API_KEY="your-api-key-here"

Usage
To run the content generation process, execute the main script and provide a topic:

python main.py "The impact of artificial intelligence on modern healthcare"

The final generated content will be saved in a designated output folder.

🤝 Contributing
We welcome contributions! Please feel free to open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
