from setuptools import setup, find_packages

# Read the contents of your README file with UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="researchbriefagent",
    version="0.1.1",
    author="Priyadharshini J",
    author_email="priyajayaraman1908@gmail.com",
    description="A Python package for creating a research brief agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vikhram-S/research_brief_agent",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'langchain-community',       # Community tools like TavilySearchResults
        'langgraph',                 # For create_react_agent and graph utilities
        'langchain-anthropic',       # Anthropic's ChatAnthropic model
        'tavily-python',             # Tavily search API client
        'langgraph-checkpoint-sqlite' # SQLite-based checkpointing for memory
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7, <3.14",
    project_urls={
        "Bug Tracker": "https://github.com/Vikhram-S/research_brief_agent",
        "Documentation": "https://github.com/Vikhram-S/research_brief_agent",
        "Source Code": "https://github.com/Vikhram-S/research_brief_agent",
    },
)
