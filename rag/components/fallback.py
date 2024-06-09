"""Here the fallback chain is defined. This chain is used when the main chain fails to generate an answer."""
from langchain.chains.base import Chain
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def generate_fallback_chain() -> Chain:
    """Generates a fallback chain to generate answers based on retrieved documents."""
    # Preamble
    preamble = (
        """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""
    )

    # LLM
    llm = ChatCohere(model_name="command-r", temperature=0).bind(preamble=preamble)

    # Prompt
    def prompt(x: dict) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([HumanMessage(f"Question: {x['question']} \nAnswer: ")])

    # Chain
    return prompt | llm | StrOutputParser()
