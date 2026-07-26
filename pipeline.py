from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:
    """
    Runs the complete multi-agent research pipeline.

    Steps:
    1. Search Agent
    2. Reader Agent
    3. Writer Chain
    4. Critic Chain

    Returns:
        dict containing all intermediate and final outputs.
    """

    state = {}

    # =====================================================
    # Step 1 - Search Agent
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 - Search Agent")
    print("=" * 60)

    search_agent = build_search_agent()

    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find recent, reliable, and detailed information about: {topic}",
                )
            ]
        }
    )

    state["search_results"] = search_result["messages"][-1].content

    print("\nSearch Results:\n")
    print(state["search_results"])

    # =====================================================
    # Step 2 - Reader Agent
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 - Reader Agent")
    print("=" * 60)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Based on the following search results about "{topic}", identify the most relevant URL.

Scrape that URL and extract the important information.

Search Results:

{state["search_results"][:1000]}
""",
                )
            ]
        }
    )

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nScraped Content:\n")
    print(state["scraped_content"])

    # =====================================================
    # Step 3 - Writer Chain
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 3 - Writer Chain")
    print("=" * 60)

    research_material = f"""
SEARCH RESULTS

{state["search_results"]}

--------------------------------------------

SCRAPED CONTENT

{state["scraped_content"]}
"""

    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research_material,
        }
    )

    print("\nFinal Research Report:\n")
    print(state["report"])

    # =====================================================
    # Step 4 - Critic Chain
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 4 - Critic Chain")
    print("=" * 60)

    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"],
        }
    )

    print("\nCritic Feedback:\n")
    print(state["feedback"])

    print("\n" + "=" * 60)
    print("Research Pipeline Completed Successfully")
    print("=" * 60)

    return state


if __name__ == "__main__":

    topic = input("Enter a research topic: ").strip()

    if topic:
        run_research_pipeline(topic)
    else:
        print("Please enter a valid research topic.")