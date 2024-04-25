from pathlib import Path

from dotenv import load_dotenv
import os
import re
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser, SemanticSplitterNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import download_loader

load_dotenv()


def clean_up_text(content: str) -> str:
    """
    Remove unwanted characters and patterns in text input.

    :param content: Text input.

    :return: Cleaned version of original text input.
    """

    # Fix hyphenated words broken by newline
    content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

    # Remove specific unwanted patterns and characters
    unwanted_patterns = [
        "\\n", "  —", "——————————", "—————————", "—————",
        r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
    ]
    for pattern in unwanted_patterns:
        content = re.sub(pattern, "", content)

    # Fix improperly spaced hyphenated words and normalize whitespace
    content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
    content = re.sub(r'\s+', ' ', content)

    return content

if __name__ == "__main__":
    print("Going to ingest llama-index documentation to pinecone")

    # Initialising our openAI llm
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.llm = llm

    # Download and instantiate `PDFReader` from LlamaHub
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()

    # Load Training data PDF from Local File System
    documents = loader.load_data(file=Path('./data/LLMCFGTrainData.pdf'))

    # Clean the documents
    cleaned_docs = []
    for d in documents:
        cleaned_text = clean_up_text(d.text)
        d.text = cleaned_text
        cleaned_docs.append(d)

    # This will be the model we use both for Node parsing and for vectorization
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    # Pinecone initialisation using api key
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Connecting to index created on pinecone
    index_name = "llamaindex-document-helper"
    pinecone_index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Define the initial pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model,
            ),
            embed_model,
        ],
        vector_store=vector_store
    )

    pipeline.run(documents=cleaned_docs)
    pinecone_index.describe_index_stats()

    print("Finished Ingestingg...")
