import json
from langchain_groq import ChatGroq
from src.workflow.tools import ResearchTools
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from src.embed.vector_db import MilvusManager
from config import Config
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

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
            max_iterations=3
        )

    def create_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate.from_template(
        """You are a helpful and conversational research assistant that helps users find and understand research papers.

        IMPORTANT: Not every response needs a tool! Only use tools when you need to fetch new information.
        - For greetings, questions about capabilities, or clarifications: Respond with just "Thought" and "Final Answer"
        - For processing or explaining existing results: Use your knowledge to structure and explain
        - For getting new information about papers: Use the complete tool format

        Previous conversation:
        {chat_history}

        You have access to these tools:
        {tools}

        IMPORTANT - Maintain Context:
        - Keep track of papers mentioned in the conversation
        - Remember arxiv_ids and titles for follow-up questions
        - When user refers to a paper by title or context, use the stored arxiv_id
        - Present information in a natural, conversational way

        Tool Usage Workflow:
        1. For initial paper searches:
           - Use get_related_papers
           - Process results to match user's needs
           - Store paper details for follow-up questions
           
        2. For paper-specific information:
           - Extract arxiv_id from context if paper was previously mentioned
           - Use get_citations for impact metrics
           - Use get_summary for detailed content
           - Use get_authors for researcher information
           - Use get_github_repo for implementation code

        Example Natural Conversation:

        User: "Find papers about transformers"
        Thought: Need to search for papers
        Action: get_related_papers
        Action Input: "transformers"
        Observation: [List of papers including "ViT: An Image is Worth 16x16 Words" with arxiv_id: 2010.11929]
        Thought: Got the papers, will present them clearly
        Final Answer: I found several papers about transformers. The most relevant one is "ViT: An Image is Worth 16x16 Words"...

        User: "Can you summarize that paper?"
        Thought: User wants a summary of the ViT paper I just mentioned. I have its arxiv_id from previous results
        Action: get_summary
        Action Input: "2010.11929"

        Remember:
        - Be conversational and helpful
        - Maintain context from previous messages
        - Store paper details for follow-up queries
        - Use natural language to transition between topics
        - For simple responses, just use Thought and Final Answer

        Begin!

        Human: {input}
        Assistant: Let me help you with that.

        {agent_scratchpad}""",
        partial_variables={"tool_names": ", ".join([tool.name for tool in self.tools])}
        )
        return prompt
    
    async def chat(self, message: str) -> str:
        """Process user message and return response"""
        try:
            response = await self.agent_executor.invoke(
                {"input": message}
            )
            return response["output"]
        except Exception as e:
            return f"I encountered an error: {str(e)}"
        

if __name__ == "__main__":
    # Initialize components
    milvus_manager = MilvusManager(CATEGORIES)
    research_tools = ResearchTools(milvus_manager)
    agent = ResearchAssistant(research_tools)
    agent_executor = agent.agent_executor
    
    print("Research Assistant initialized. Type 'quit' to exit.")
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
