from pathlib import Path

from dotenv import load_dotenv
import os
import re
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
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

def filename_fn(filename):
    if filename:
        file_name = Path(filename).name
        return {"file_name": file_name}
    else:
        return {"file_name": "unknown"}


if __name__ == "__main__":
    print("Going to ingest llama-index documentation to pinecone")

    # Download and instantiate `PDFReader` from LlamaHub
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()

    # Load Training data PDF from Local File System
    documents = loader.load_data(file=Path('./data/llama2.pdf'))

    # Clean the documents
    cleaned_docs = []
    for d in documents:
        cleaned_text = clean_up_text(d.text)
        d.text = cleaned_text
        cleaned_docs.append(d)

    from llama_index.readers.file import UnstructuredReader

    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs-tmp",
        file_extractor={".html": UnstructuredReader()},
        file_metadata=filename_fn,
    )

    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # Initialising our openAI llm
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    # Initialising our EmbedModel
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    # Pinecone initialisation using api key
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Connecting to index created on pinecone
    index_name = "llamaindex-document-helper"
    pinecone_index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Injecting our llm, embed model and node parser are injected to Llama-Index settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print("Finished Ingestingg...")
