# Research Brief Agent

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/research_brief_agent)](https://pypi.org/project/research_brief_agent/)
[![PyPI - License](https://img.shields.io/pypi/l/research_brief_agent)](https://pypi.org/project/research_brief_agent/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Vikhram-S/research_brief_agent/commits/main)
[![PyPI](https://img.shields.io/pypi/v/research_brief_agent)](https://pypi.org/project/research_brief_agent/)
[![PyPI - Status](https://img.shields.io/pypi/status/research_brief_agent)](https://pypi.org/project/research_brief_agent/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/research_brief_agent)](https://pypi.org/project/research_brief_agent/)
[![Total Downloads](https://pepy.tech/badge/research_brief_agent)](https://pepy.tech/project/research_brief_agent)

A Python library for an AI agent that generates concise research briefs from web searches using LangChain, Anthropic's Claude, and Tavily search.

## Overview

The `ResearchBriefAgent` automates the process of researching a topic and producing a structured brief, including key points and sources. Built with LangChain's ecosystem, it leverages Anthropic's Claude model for reasoning and Tavily for web search, with SQLite-backed memory for context persistence.

## Features

- **Topic-Based Research**: Generate briefs on any topic with a single function call.
- **Structured Output**: Receive formatted briefs with key points and cited sources.
- **Memory Persistence**: Retain conversation context across queries using SQLite checkpointing.
- **Extensible**: Built on LangChain, allowing integration with additional tools and models.

## Installation

Install the library via PyPI:

```bash
pip install research_brief_agent
```
# Usage
```
from research_brief_agent import ResearchBriefAgent
agent = ResearchBriefAgent(anthropic_api_key="your_anthropic_api_key", tavily_api_key="your_tavily_api_key")
brief = agent.generate_brief("Impact of renewable energy on climate change")
print(brief)
```
Example Output:
## Research Brief: Impact of renewable energy on climate change
### Key Points
- Reduces greenhouse gas emissions by replacing fossil fuels.
- Solar and wind power adoption has surged globally.
- Critical for limiting warming to 1.5°C.
### Sources
- https://www.iea.org/reports/renewables-2023
- https://www.un.org/en/climatechange/renewable-energy
## Requirements
Python: 3.8 or higher
Dependencies: langchain-community, langgraph, langchain-anthropic, tavily-python, langgraph-checkpoint-sqlite
API Keys: Obtain from Anthropic and Tavily
Setup
Clone the repo:
```
git clone https://github.com/yourusername/research_brief_agent.git
cd research_brief_agent
```
Install dependencies:
```
pip install -r requirements.txt
```

Or install from PyPI:
```
pip install research_brief_agent
```

# Contributing
Contributions are welcome! Fork the repo, create a feature branch, commit changes, and open a pull request. See CONTRIBUTING.md for details (to be added).

# License
MIT License - see LICENSE for details.

# Contact
Email: your.email@example.com

Issues: GitHub

# Acknowledgments
Built with LangChain and Anthropic.
Search powered by Tavily.

---

### Verification
This is the exact content from my previous response, presented as a single Markdown file. You can save it directly as `README.md` in your project directory. The "Usage" section remains a single, copy-paste-friendly block, and all badges, sections, and details are preserved.

### Next Steps
- Replace placeholders (`yourusername`, `your.email@example.com`) with your actual details.
- Save this text as `README.md` in your project root.
- Push to GitHub and publish to PyPI to activate the badges.

Let me know if you need help with any part of the implementation!