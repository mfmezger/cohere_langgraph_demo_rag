"""Here the grader chains are defined. These chains are used to grade the quality of the generated answers, documents and halluzinations."""
from langchain.core import chain
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Data model
def generate_document_grader() -> chain:
    """Generates a grader chain to assess relevance of a retrieved document to a user question.

    Returns
    -------
        chain: Grader chain to assess relevance of a retrieved document to a user question.
    """

    class GradeDocuments(BaseModel):

        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    # Prompt
    preamble = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments, preamble=preamble)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader


def generate_hallucination_grader() -> chain:
    """Generates a grader chain to assess hallucination in a generation answer.

    Returns
    -------
        chain: Chain to assess hallucination in a generation answer.
    """

    # Data model
    class GradeHallucinations(BaseModel):

        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    # Preamble
    preamble = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations, preamble=preamble)

    # Prompt
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            # ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    return hallucination_prompt | structured_llm_grader


def generate_answer_grader() -> chain:
    """Generates a grader chain to assess whether an answer addresses a question.

    Returns
    -------
        chain: Chain to assess whether an answer addresses a question.
    """

    class GradeAnswer(BaseModel):

        """Binary score to assess answer addresses question."""

        binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    # Preamble
    preamble = """You are a grader assessing whether an answer addresses / resolves a question \n
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer, preamble=preamble)

    # Prompt
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader
