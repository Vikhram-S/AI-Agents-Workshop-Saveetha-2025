# research_brief_agent/research_brief_agent/agent.py
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

class ResearchBriefAgent:
    def __init__(self, anthropic_api_key: str, tavily_api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize the research brief agent with API keys and model configuration.
        
        Args:
            anthropic_api_key (str): API key for Anthropic's Claude model.
            tavily_api_key (str): API key for Tavily search.
            model (str): Anthropic model name (default: "claude-3-5-sonnet-20240620").
        """
        # Initialize the language model
        self.llm = ChatAnthropic(
            model=model,
            api_key=anthropic_api_key
        )
        
        # Initialize the search tool
        self.search_tool = TavilySearchResults(api_key=tavily_api_key, max_results=5)
        
        # Set up memory for conversation context
        self.memory = MemorySaver()
        
        # Create the ReAct agent with tools and memory
        self.agent = create_react_agent(
            model=self.llm,
            tools=[self.search_tool],
            checkpointer=self.memory
        )
    
    def generate_brief(self, topic: str, thread_id: str = "default_thread") -> str:
        """
        Generate a research brief for the given topic.
        
        Args:
            topic (str): The research topic (e.g., "Impact of renewable energy").
            thread_id (str): Identifier for conversation thread (default: "default_thread").
        
        Returns:
            str: A formatted research brief with key points and sources.
        """
        # Define the prompt for the agent
        prompt = (
            f"Generate a concise research brief on '{topic}'. "
            "Use the search tool to gather information. "
            "Format the output as:\n"
            "## Research Brief: {topic}\n"
            "### Key Points\n- Point 1\n- Point 2\n- Point 3\n"
            "### Sources\n- Source 1\n- Source 2"
        )
        
        # Prepare the input message
        message = HumanMessage(content=prompt)
        
        # Configure the thread for memory persistence
        config = {"configurable": {"thread_id": thread_id}}
        
        # Execute the agent and get the response
        response = self.agent.invoke({"messages": [message]}, config)
        
        # Extract the agent's response content
        for msg in response["messages"][::-1]:  # Get the latest AI message
            if msg.type == "ai":
                return msg.content
        return "Failed to generate research brief."