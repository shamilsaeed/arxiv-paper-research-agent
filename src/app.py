import streamlit as st
from src.workflow.agent import ResearchAssistant
from src.workflow.tools import ResearchTools
from src.embed.vector_db import MilvusManager

def initialize_agent():
    """Initialize the research assistant"""
    if 'agent' not in st.session_state:
        milvus_manager = MilvusManager()
        research_tools = ResearchTools(milvus_manager)
        st.session_state.agent = ResearchAssistant(research_tools)
        st.session_state.messages = []

def process_message(user_input: str):
    """Process user message and update chat history"""
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get agent response
        response = st.session_state.agent.agent_executor.invoke(
            {"input": user_input}
        )
        
        # Clean the response
        output = response["output"]
        if "Final Answer:" in output:
            output = output.split("Final Answer:")[-1].strip()
        elif "Thought:" in output:
            thought_parts = output.split("Thought:")
            output = thought_parts[-1].strip()
            if "Action:" in output:
                output = output.split("Action:")[0].strip()
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": output})

def main():
    st.title("ArXiv Research Assistant")
    st.write("Ask me about research papers!")
    
    # Initialize agent
    initialize_agent()
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    # Process new message
    if user_input:
        process_message(user_input)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main() 