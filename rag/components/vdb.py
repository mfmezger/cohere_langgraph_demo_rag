"""Here the vectorstore is defined. This vectorstore is used to store and retrieve documents."""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core import chain


def load_vdb_retriver() -> chain:
    """Load the vectorstore retriever."""
    # Set embeddings
    embd = CohereEmbeddings()

    # Docs to index
    urls = [
        "https://www.stackit.de/en/general-terms-and-conditions/service-certificates/stackit-compute-engine-gpu/",
        "https://docs.stackit.cloud/stackit/en/how-to-install-nvidia-gpu-drivers-165511643.html",
        "https://docs.stackit.cloud/stackit/en/faq-known-issues-of-ske-28476393.html",
    ]

    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embd,
    )

    return vectorstore.as_retriever()
