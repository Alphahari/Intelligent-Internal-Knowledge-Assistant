# 🧠 NLP Tool-Calling Agent with LangGraph, Groq, and LangChain

This project demonstrates an intelligent NLP agent capable of answering questions using multiple tools, including PDF parsing, text file reading, and Slack message retrieval. It utilizes **LangGraph**, **LangChain**, and **Groq's Qwen model** for tool-aware conversational reasoning.

---

## 🚀 Features

- 📄 **PDF Tool** – Search for content inside `attention.pdf`
- 📜 **Text Tool** – Read from a speech stored in `speech.txt`
- 💬 **Slack Tool** – Retrieve the latest messages from a Slack channel
- 🧠 **LLM** – Uses Groq's `qwen-qwq-32b` model via LangChain
- ⚙️ **Tool Calling** – Dynamic function calling with LangGraph and LangChain tool integration

---

## 🏗️ Project Structure

```bash
nlp_project/
│
├── main.py                  # Entry point to the tool-calling agent
├── requirements.txt         # Python dependencies
├── .env                     # Contains GROQ API key
│
├── attention.pdf            # Sample paper used by PDF tool
├── speech.txt               # Example text content
│
├── tools/
│   ├── document_loader.py   # Text_Document tool
│   ├── pdf_tool.py          # PDF loading/searching tool
│   ├── slack.py             # Slack message retriever tool
│
└── utils/                   # (Optional) Utility functions
```

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Alphahari/nlp_project.git
cd nlp_project
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv agenticAI
agenticAI\Scripts\activate    # On Windows
source agenticAI/bin/activate  # On Mac/Linux

pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:
```env
GROQ_API=your_groq_api_key_here
```

## 🧪 Run the Agent
```bash
python main.py
```
You should see:
- A tool call graph visualization
- Tool usage printed in the terminal
- The final answer based on queried documents

## 📚 Example Query
```text
Authors of Attention is all you need
```
The agent will:
- Detect the need for document lookup.
- Call the PDF tool.
- Extract and return relevant author information from attention.pdf.

## 🛠️ Dependencies

- langgraph
- langchain
- langchain_groq
- python-dotenv
- IPython
- groq
- Custom tools under `tools/`

## 🧩 Extending This Project

- Add more tools (e.g., web search, database query)
- Swap out Groq for another LLM provider (OpenAI, Anthropic)
- Add memory or history tracking for multi-turn dialogues
- Integrate with a web frontend using Streamlit or Gradio

## 📜 License

This project is licensed under the MIT License.

## 👤 Author

**Hari Krishna Reddy**  
GitHub: [Alphahari](https://github.com/Alphahari)
