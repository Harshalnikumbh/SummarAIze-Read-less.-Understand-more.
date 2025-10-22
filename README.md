---
title: SummarAIze
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# SummarAIze 🤖✨

**AI-Powered Summarization for E-commerce Reviews & Web Pages**

SummarAIze is an intelligent text summarization application that transforms lengthy product reviews and web content into concise, human-readable summaries.

## 🎯 Features

- **E-commerce Review Summarization**: Aggregate and summarize multiple product reviews
- **Web Page Summarization**: Extract and summarize content from any URL
- **Dual-Model Architecture**: Combines T5 and TinyLlama for optimal results
- **Human-like Output**: Natural language summaries that are easy to read

## 🔧 How It Works

SummarAIze uses a sophisticated three-stage pipeline:

1. **Web Scraping**: Extracts content from URLs using BeautifulSoup
2. **Initial Summarization**: T5 model processes and condenses the raw text
3. **Natural Language Refinement**: TinyLlama transforms the T5 output into human-like, conversational summaries

## 🚀 Technology Stack

- **Backend**: Flask
- **ML Models**: 
  - T5 (Text-to-Text Transfer Transformer)
  - TinyLlama (Language Model)
- **Web Scraping**: BeautifulSoup4
- **Deployment**: Docker

## 💡 Use Cases

- Quickly understand product sentiment from hundreds of reviews
- Get the gist of long articles and blog posts
- Research competitor products efficiently
- Save time on content consumption

## 🛠️ Local Development

```bash
# Clone the repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📦 Docker Deployment

```bash
# Build the Docker image
docker build -t summaraize .

# Run the container
docker run -p 7860:7860 summaraize
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License.

---

Built with ❤️ using Flask, T5, and TinyLlama