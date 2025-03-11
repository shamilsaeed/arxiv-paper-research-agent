import json

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import Config
from src.embed.vector_db import MilvusManager
from src.workflow.tools import ResearchTools

CATEGORIES = json.load(open("categories.json"))

config = Config()
MODEL = config.groq.model
GROQ_API_KEY = config.groq.api_key

class ResearchAssistant:
    def __init__(self, research_tools: ResearchTools):
        # Initialize LLM
        self.llm = ChatGroq(
            model=MODEL,  
            temperature=0,
            groq_api_key=GROQ_API_KEY 
        )
        
        # Get tools from ResearchTools
        self.tools = research_tools.tools
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create ReAct agent with memory
        prompt = self.create_prompt()
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Create executor with memory
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,  # Add memory to executor
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def create_prompt(self):
        prompt = PromptTemplate.from_template(
        """You are a research assistant helping users find and understand academic papers.

        You have these tools:
        {tools}

        IMPORTANT: Only use tools when you need to fetch information about papers!
        For simple responses (no tools needed):
        1. Write your Thought
        2. Write Final Answer immediately after

        For tool usage:
        1. Write your Thought
        2. Write Action and Action Input
        3. Get Observation
        4. Write Final Answer

        Tool Usage Guide:

        1. get_related_papers: Initial search tool
           - USE FOR: Finding papers on a topic, authors, publish dates, pdf url
           - INPUT: Search query (e.g., "transformers in vision")
           - RETURNS: List of papers with metadata

        2. get_paper_processed: Preparation for deep paper analysis
           - USE FOR: When user wants to understand paper content
           - INPUT: arxiv_id (e.g., "2103.14030")
           - MUST USE before get_paper_details or get_summary

        3. get_paper_details: RAG-based Q&A
           - USE FOR: Specific questions about paper content
           - INPUT: query and arxiv_id (e.g. query = "What is the experimental setup used in the paper?", arxiv_id = "2103.14030")
           - REQUIRES: Paper must be processed first

        4. get_summary: Generate paper summary
           - USE FOR: Overview of paper content
           - INPUT: arxiv_id (e.g., "2103.14030")
           - REQUIRES: Paper must be processed first

        5. get_citations: Citation metrics
           - USE FOR: Paper impact analysis
           - INPUT: arxiv_id (e.g., "2103.14030")

        6. get_github_repo: Find code
           - USE FOR: Implementation/code requests
           - INPUT: Paper title (e.g. "Transformer Models in Computer Vision")

        Previous conversation:
        {chat_history}

        IMPORTANT RULES:
        1. For greetings or general chat: Just respond with Thought then Final Answer
        2. Keep track of the arxiv_id of the papers from chat history so you can use them to further answer questions
        2. Only use tools for paper-related queries
        3. Your available actions are: [{tool_names}]

        Examples:

        Human: "Hi there!"
        Thought: This is a greeting, so I'll respond directly without using any tools.
        Final Answer: Hello! I'm your research assistant. How can I help you find or understand research papers today?

        Human: "Find me papers about transformer models in computer vision?"
        Thought: Need to search for relevant papers using the search tool.
        Action: get_related_papers
        Action Input: "transformer models in computer vision"
        Observation: [paper 1 with metadata, paper 2 with metadata, ...] Let me present them in bullet points.
        Final Answer: I found several interesting papers that will be useful for your research.

        Human: "For the first paper, who are the authors?"
        Thought: Since this is metadata, I can check previous responses to identify the authors of this paper.
        Action: get_related_papers
        Action Input: "LLMs in natural language processing"
        Observation: [papers with metadata]
        Final Answer: I found several interesting papers about LLMs...

        Human: "Summarize the paper relating to ViT: Transformers"
        Thought: I need to process the paper first, then get its summary.
        Action: get_paper_processed
        Action Input: 2407.1451
        Observation: True
        Thought: Now I can get the paper summary.
        Action: get_summary
        Action Input: 2407.1451
        Observation: [Summary of the paper...]
        Thought: I now have the complete summary to share with the user.
        Final Answer: Here is a summary of the paper
        [Insert formatted summary here]
        
        IMPORTANT FORMAT RULES:
        - NEVER combine Action and Final Answer in the same response
        - Complete ALL necessary actions before giving Final Answer
        - Each response should be either:
          1. Thought → Action → Action Input
          OR
          2. Thought → Final Answer
        - After an Observation, always start with a new Thought
        
        Begin!

        Human: {input}
        Assistant: Let me help you with that.

        {agent_scratchpad}""",
        partial_variables={"tool_names": ", ".join([tool.name for tool in self.tools])}
        )
        return prompt

if __name__ == "__main__":
    # Initialize components
    milvus_manager = MilvusManager()
    research_tools = ResearchTools(milvus_manager)
    agent = ResearchAssistant(research_tools)
    agent_executor = agent.agent_executor
    
    print("ArxiV Research Assistant initialized. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            # Get agent response
            print("\nAssistant:", end=' ')
            response = agent_executor.invoke({"input": user_input})
            
            # Extract the actual response, whether from Final Answer or Thought
            output = response["output"]
            if "Final Answer:" in output:
                output = output.split("Final Answer:")[-1].strip()
            elif "Thought:" in output:
                # If there's no Final Answer but there is a Thought
                thought_parts = output.split("Thought:")
                output = thought_parts[-1].strip()  # Get the last thought
                # Remove any Action/Observation markers if present
                if "Action:" in output:
                    output = output.split("Action:")[0].strip()
            
            # Clean up any remaining markers
            output = output.replace("Invalid Format:", "").strip()
            
            print(output)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")
