"""Chunk post-processing helpers applied after chunking.

Concrete processors are used through the :class:`ChunkProcessor` protocol and
should rely on structural typing rather than inheriting from the protocol.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from grawiki.core.commons import Chunk


class ChunkProcessor(Protocol):
    """Protocol for asynchronously processing chunks of text.

    Implementations should return a new :class:`~grawiki.core.commons.Chunk`
    rather than mutating the input in place.
    """

    async def __call__(self, chunk: Chunk) -> Chunk:
        """Process a chunk of text.

        Parameters
        ----------
        chunk : Chunk
            The chunk of text to process.

        Returns
        -------
        Chunk
            The processed chunk of text.
        """


class HypotheticalQuestions(BaseModel):
    """Structured hypothetical questions generated from a chunk.

    Parameters
    ----------
    questions : list[str], optional
        Generated questions associated with the chunk.
    """

    questions: list[str] = Field(default_factory=list)


class HypotheticalQuestionsChunkProcessor:
    """Generate hypothetical questions and prepend them to chunk content.

    Parameters
    ----------
    model : str
        Model name used by :class:`pydantic_ai.Agent`.
    num_question : int, optional
        Maximum number of questions to request from the model. Defaults to ``3``.
    language : str, optional
        Language to use for generated questions. Defaults to ``"english"``.
    *args
        Additional positional arguments forwarded to :class:`pydantic_ai.Agent`.
    **kwargs
        Additional keyword arguments forwarded to :class:`pydantic_ai.Agent`.
    """

    QUESTIONS_GENERATION_PROMPT = """Given a chunk of text, generate a hypothetical question that could be asked about the content of the chunk. 
    The question should be strictly relevant to the content. Questions should be concise and clear.
    Generate max {num_question} questions in {language} language - regardless of the original language of the document."""

    def __init__(
        self,
        model: str,
        num_question: int = 3,
        language: str = "english",
        *args,
        **kwargs,
    ) -> None:
        """Initialize the processor and its backing structured-output agent.

        Parameters
        ----------
        model : str
            Model name used by :class:`pydantic_ai.Agent`.
        num_question : int, optional
            Maximum number of questions to request. Defaults to ``3``.
        language : str, optional
            Output language for generated questions. Defaults to ``"english"``.
        *args
            Additional positional arguments forwarded to :class:`pydantic_ai.Agent`.
        **kwargs
            Additional keyword arguments forwarded to :class:`pydantic_ai.Agent`.
        """

        self.model = model
        self.num_question = num_question
        self.language = language
        self.agent = Agent(
            model=model,
            system_prompt=self.QUESTIONS_GENERATION_PROMPT.format(
                num_question=num_question,
                language=language,
            ),
            output_type=HypotheticalQuestions,
            *args,
            **kwargs,
        )

    def format_agent_response(self, response: HypotheticalQuestions) -> str:
        """Format generated questions for inclusion in chunk content.

        Parameters
        ----------
        response : HypotheticalQuestions
            Structured question output returned by the agent.

        Returns
        -------
        str
            Formatted question block, or an empty string when the model returns
            no questions.
        """

        if not response.questions:
            return ""

        lines = [
            "--- HYPOTHETICAL QUESTIONS IN THIS CHUNK ---",
            *(f"- {question}" for question in response.questions),
            "---",
        ]
        return "\n".join(lines)

    async def __call__(self, chunk: Chunk) -> Chunk:
        """Return a processed copy of ``chunk`` with prepended questions.

        Parameters
        ----------
        chunk : Chunk
            Source chunk to enrich.

        Returns
        -------
        Chunk
            New chunk instance with copied metadata and updated content. When the
            agent returns no questions, the content is preserved unchanged.
        """

        chunk_content = chunk.content
        questions = await self.agent.run(chunk_content)
        formatted_result = self.format_agent_response(questions.output)
        new_content = (
            f"{formatted_result}\n\n{chunk_content}"
            if formatted_result
            else chunk_content
        )
        return chunk.model_copy(
            deep=True,
            update={"content": new_content, "metadata": dict(chunk.metadata)},
        )
