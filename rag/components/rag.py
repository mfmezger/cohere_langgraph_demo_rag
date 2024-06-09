"""Here the RAG chain is defined. This chain is used to generate answers based on retrieved documents."""
from langchain_cohere import ChatCohere
from langchain_core import chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def generate_rag_chain() -> chain:
    """Generates a RAG chain to generate answers based on retrieved documents."""
    # Preamble
    preamble = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

    # LLM
    llm = ChatCohere(model_name="command-r", temperature=0).bind(preamble=preamble)

    # Prompt
    def prompt(x: dict) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([HumanMessage(f"Question: {x['question']} \nAnswer: ", additional_kwargs={"documents": x["documents"]})])

    # Chain
    return prompt | llm | StrOutputParser()
