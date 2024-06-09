"""Here the router chain is defined. This chain is used to route a user question to a vectorstore or web search."""
from langchain_cohere import ChatCohere
from langchain_core import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


def generate_question_router() -> chain:
    """Generates a router chain to route a user question to a vectorstore or web search."""

    # Data model
    class WebSearch(BaseModel):

        """The internet. Use web_search for questions that are related to anything else than related to stackits information about nvidia installation, gpu virtualmaschines and the general cloud plattform faq."""

        query: str = Field(description="The query to use when searching the internet.")

    class Vectorstore(BaseModel):

        """A vectorstore containing documents related to stackits information about nvidia installation, gpu virtualmaschines and the general cloud plattform faq. Use the vectorstore for questions on these topics."""

        query: str = Field(description="The query to use when searching the vectorstore.")

    # Preamble
    preamble = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to stackits information about nvidia installation, gpu virtualmaschines and the general cloud plattform faq.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""

    # LLM with tool use and preamble
    llm = ChatCohere(model="command-r", temperature=0)
    structured_llm_router = llm.bind_tools(tools=[WebSearch, Vectorstore], preamble=preamble)

    # Prompt
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
        ]
    )

    return route_prompt | structured_llm_router
