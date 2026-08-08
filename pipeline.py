"""Multi-agent research pipeline: search -> read -> write -> critique."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

logger = logging.getLogger(__name__)

# Callback signature: (step_name, status) -> None
# step_name is one of "search" | "reader" | "writer" | "critic"
# status is one of "running" | "done" | "error"
ProgressCallback = Optional[Callable[[str, str], None]]


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails in a way the pipeline can't recover from."""


@dataclass
class ResearchState:
    topic: str
    search_results: str = ""
    scraped_content: str = ""
    report: str = ""
    feedback: str = ""
    errors: List[str] = field(default_factory=list)


def _notify(callback: ProgressCallback, step: str, status: str) -> None:
    if callback:
        callback(step, status)


def run_research_pipeline(topic: str, progress_callback: ProgressCallback = None) -> ResearchState:
    """
    Run the full research pipeline for a topic.

    Steps:
        1. Search Agent   - gathers recent web information
        2. Reader Agent   - scrapes the most relevant URL for deeper content
        3. Writer Chain   - drafts a structured report
        4. Critic Chain   - reviews and scores the report

    Steps 1 and 3 are fatal if they fail (there's nothing downstream can do
    without them). Steps 2 and 4 degrade gracefully: a failed scrape falls
    back to search-result summaries, and a failed critique just means no
    feedback is attached to an otherwise-complete report.

    Args:
        topic: The research topic/question.
        progress_callback: Optional callback(step_name, status) invoked as
            each step starts, finishes, or errors — useful for driving a
            live UI (see app.py).

    Returns:
        ResearchState with all intermediate and final outputs.

    Raises:
        ValueError: if topic is empty.
        PipelineError: if a fatal step (search or writer) fails.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    state = ResearchState(topic=topic.strip())

    # ---- Step 1: Search (fatal on failure) ----
    _notify(progress_callback, "search", "running")
    try:
        search_agent = build_search_agent()
        result = search_agent.invoke(
            {"messages": [("user", f"Find recent, reliable, and detailed information about: {state.topic}")]}
        )
        state.search_results = result["messages"][-1].content
        logger.info("Search step complete (%d chars)", len(state.search_results))
        _notify(progress_callback, "search", "done")
    except Exception as exc:
        logger.exception("Search step failed")
        state.errors.append(f"Search step failed: {exc}")
        _notify(progress_callback, "search", "error")
        raise PipelineError(f"Search step failed: {exc}") from exc

    # ---- Step 2: Read (non-fatal on failure) ----
    _notify(progress_callback, "reader", "running")
    try:
        reader_agent = build_reader_agent()
        result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{state.topic}', identify the "
                        f"most relevant URL, scrape it, and extract the important information.\n\n"
                        f"Search Results:\n{state.search_results[:1500]}",
                    )
                ]
            }
        )
        state.scraped_content = result["messages"][-1].content
        logger.info("Reader step complete (%d chars)", len(state.scraped_content))
        _notify(progress_callback, "reader", "done")
    except Exception as exc:
        logger.exception("Reader step failed; continuing with search results only")
        state.errors.append(f"Reader step failed: {exc}")
        state.scraped_content = "(No additional content could be scraped due to an error.)"
        _notify(progress_callback, "reader", "error")

    # ---- Step 3: Write (fatal on failure) ----
    _notify(progress_callback, "writer", "running")
    research_material = (
        f"SEARCH RESULTS\n\n{state.search_results}\n\n"
        f"{'-' * 40}\n\nSCRAPED CONTENT\n\n{state.scraped_content}"
    )
    try:
        state.report = writer_chain.invoke({"topic": state.topic, "research": research_material})
        logger.info("Writer step complete (%d chars)", len(state.report))
        _notify(progress_callback, "writer", "done")
    except Exception as exc:
        logger.exception("Writer step failed")
        state.errors.append(f"Writer step failed: {exc}")
        _notify(progress_callback, "writer", "error")
        raise PipelineError(f"Writer step failed: {exc}") from exc

    # ---- Step 4: Critique (non-fatal on failure) ----
    _notify(progress_callback, "critic", "running")
    try:
        state.feedback = critic_chain.invoke({"report": state.report})
        logger.info("Critic step complete")
        _notify(progress_callback, "critic", "done")
    except Exception as exc:
        logger.exception("Critic step failed")
        state.errors.append(f"Critic step failed: {exc}")
        state.feedback = "(Critic review unavailable due to an error.)"
        _notify(progress_callback, "critic", "error")

    return state


if __name__ == "__main__":
    topic_input = input("Enter a research topic: ").strip()

    if not topic_input:
        print("Please enter a valid research topic.")
    else:
        try:
            final_state = run_research_pipeline(
                topic_input,
                progress_callback=lambda step, status: print(f"[{step}] {status}"),
            )
        except PipelineError as e:
            print(f"\nPipeline failed: {e}")
        else:
            print("\n" + "=" * 60)
            print("FINAL REPORT")
            print("=" * 60)
            print(final_state.report)
            print("\n" + "=" * 60)
            print("CRITIC FEEDBACK")
            print("=" * 60)
            print(final_state.feedback)
            if final_state.errors:
                print("\n" + "=" * 60)
                print("NON-FATAL WARNINGS")
                print("=" * 60)
                for err in final_state.errors:
                    print(f"- {err}")