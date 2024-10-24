import streamlit as st
from chain_setup import conversational_rag_chain
import markdown2
import pyperclip
from st_copy_to_clipboard import st_copy_to_clipboard

# Title for the app
st.title("AI Assistant")

# Initialize session state for storing chat history if not already initialized
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize session state for storing sources if not already initialized
if 'sources' not in st.session_state:
    st.session_state.sources = []

# Initialize session state for user input if not already initialized
if 'submitted_input' not in st.session_state:
    st.session_state.submitted_input = ""

# Function to invoke the conversational chain and update the chat history
def process_message():
    user_input = st.session_state.get('submitted_input', '')  # Get the submitted input
    if user_input.strip():
        # Add user message to chat history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Call your RAG chain 
        try:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": "user"}
                }
            )
            st.session_state.sources = list(set([document.metadata['source'] for document in response["context"]]))
            
            # Convert AI's response (in markdown format) to plain markdown (with LaTeX support)
            ai_response_markdown = response.get('answer', '') 
            print(ai_response_markdown)
            
            # Add AI response to chat history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": ai_response_markdown,
                "sources": st.session_state.sources
            })

        except Exception as e:
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": "Error generating response.",
                "sources": "Null"
            })
            print(f"Error occurred: {e}")
        
        # Clear the input after sending the message
        st.session_state.submitted_input = ""  # Clear the submitted input

# Define the style for the chat display
style = """
    <style>
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    </style>
"""

# Display conversation history with containers and a copy button for each message
for idx, message in enumerate(st.session_state.conversation_history):
    with st.container():
        role = message['role']
        if role == "user":
            title = "<br><strong>User:</strong>"
            message_html = f'<div style="background-color: #D6E4FF; padding: 10px; color: #000000; border-radius: 10px;">\n {message["content"]}</div>'
        elif role == "assistant":
            title = "<br><strong>AI Assistant:</strong> Sources: {}".format(" | ".join(message['sources']))
            message_html = f'<div style="background-color: #F5F5F5; padding: 10px; border-radius: 10px; color: #000000;">\n {message["content"]}</div>'
        
        # Display the message with custom styling
        st.markdown(title, unsafe_allow_html=True)
        st.markdown(f'{style}{message_html}', unsafe_allow_html=True)
        if role == "assistant":
           st_copy_to_clipboard(message["content"])

# Use a text_area for user input and store its value in 'submitted_input'
user_input = st.text_area(
    "You:", 
    value="", 
    placeholder="Type your message here...", 
    height=100,
)

# Button to send the message
if st.button("Send"):
    st.session_state.submitted_input = user_input
    process_message()
    st.rerun()
# Display the sources in a readable format
st.write("Sources:")
for source in st.session_state.sources:
    st.write(f"- {source}")
