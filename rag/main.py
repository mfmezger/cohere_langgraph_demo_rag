"""Main RAG Graph."""

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from rag.components.fallback import generate_fallback_chain
from rag.components.grader import generate_answer_grader, generate_document_grader, generate_hallucination_grader
from rag.components.rag import generate_rag_chain
from rag.components.router import generate_question_router
from rag.components.vdb import load_vdb_retriver

load_dotenv()

# Setup the necessary components
retriever = load_vdb_retriver()
question_router = generate_question_router()
rag_chain = generate_rag_chain()
llm_chain = generate_fallback_chain()

# define graders
retrieval_grader = generate_document_grader()
hallucination_grader = generate_hallucination_grader()
answer_grader = generate_answer_grader()


web_search_tool = TavilySearchResults()


class RouterError(Exception):

    """Router exception."""


class GraphState(TypedDict):

    """Represents the state of our graph.

    Attributes
    ----------
        question: question
        generation: LLM generation
        documents: list of documents

    """

    question: str
    generation: str
    documents: list[str]


def retrieve(state: GraphState) -> dict:
    """Retrieve documents.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        state (dict): New key added to state, documents, that contains retrieved documents

    """
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def llm_fallback(state: GraphState) -> dict:
    """Generate answer using the LLM w/o vectorstore.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        state (dict): New key added to state, generation, that contains LLM generation

    """
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}


def generate(state: GraphState) -> dict:
    """Generate answer using the vectorstore.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        state (dict): New key added to state, generation, that contains LLM generation

    """
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: GraphState) -> dict:
    """Determines whether the retrieved documents are relevant to the question.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        state (dict): Updates documents key with only filtered relevant documents

    """
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            continue
    return {"documents": filtered_docs, "question": question}


def web_search(state: GraphState) -> dict:
    """Web search based on the re-phrased question.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        state (dict): Updates documents key with appended web results

    """
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


### Edges ###


def route_question(state: GraphState) -> str:
    """Route question to web search or RAG.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        str: Next node to call

    """
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Fallback to LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        return "llm_fallback"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        msg = "Router could not decide source"
        raise RouterError(msg)

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "web_search":
        return "web_search"
    elif datasource == "vectorstore":
        return "vectorstore"
    else:
        return "vectorstore"


def decide_to_generate(state: GraphState) -> str:
    """Determines whether to generate an answer, or re-generate a question.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        str: Binary decision for next node to call

    """
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """Determines whether the generation is grounded in the document and answers question.

    Args:
    ----
        state (dict): The current graph state

    Returns:
    -------
        str: Decision for next node to call

    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        # Check question-answering
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # rag
workflow.add_node("llm_fallback", llm_fallback)  # llm

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # Hallucinations: re-generate
        "not useful": "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("llm_fallback", END)

# Compile
app = workflow.compile()

# Run
inputs = {"question": "Can you give me the necessary stepts to  install nvidia on stackit?"}

# for output in app.stream(inputs):
#     for key, value in output.items():
#         # Node
#         logger.info(f"Node '{key}':")
#         # Optional: print full state at each node
#         # logger.info(value["keys"], indent=2, width=80, depth=None)

# # Final generation
# logger.info(value["generation"])


inputs = {"question": "What is an ETF?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        logger.info(f"Node '{key}':")

# Final generation
logger.info(value["generation"])
