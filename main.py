from dotenv import load_dotenv
import os

from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode

from CFGImageGenerator import CFGImageGenerator
from prompts import PROMPT_IMPROVER

load_dotenv()
import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print("RAG..")

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])

    Settings.callback_manager = callback_manager

    pinecone_index = pc.Index(name="llamaindex-document-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = get_index()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True
    )

st.set_page_config(
    page_title="Chat with CFG Large Language Model",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with CFG LLM 🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Give me any code and I will generate the Control Flow Diagram"
        }
    ]

if prompt:= st.chat_input("Your question"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=PROMPT_IMPROVER + prompt)
            st.write(response.response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.response
            })
            cfg = CFGImageGenerator()
            print(response.response)
            cfg.generate_image(response.response)