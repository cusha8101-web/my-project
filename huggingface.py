import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables (make sure your HF API token is in .env)
load_dotenv()

# Initialize model
llmmodel = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llmmodel)

# Streamlit UI
st.title("🤖 AI Chatbot (Hugging Face + Streamlit)")

# Textbox for input
user_input = st.text_input("Ask me a question:")

# Button
if st.button("Get Answer"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = model.invoke(user_input)
            st.subheader("Answer:")
            st.write(result.content)
    else:
        st.warning("Please enter a question first.")
