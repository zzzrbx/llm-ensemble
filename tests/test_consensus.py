from textwrap import dedent
from typing import TypedDict

from rich import print
from rich.console import Console
from rich.markdown import Markdown

from llm_ensemble import Consensus

console = Console()

class UserSchema(TypedDict):
    """Output schema for sustainability consensus test."""
    consensus: bool    # Whether consensus was reached among agents
    final_answer: str  # The agreed-upon answer to the question
    notes: str         # Process insights, key points of agreement/disagreement


def test_schema():
    """
    Test Consensus with a complex sustainability governance question.

    This test explores whether mandatory sustainability disclosure creates
    genuine accountability or merely shifts responsibility from governments
    to corporations.
    """
    print("\n" + "="*80)
    print("CONSENSUS TEST - With schema")
    print("="*80)

    consensus = Consensus(
        models=[
            "openai:gpt-5-mini",
            "google_genai:gemini-3-flash-preview",
            "anthropic:claude-3-5-haiku-20241022",
            "xai:grok-3-mini",
        ],
        response_schema=UserSchema
    )

    prompt = (
        "If survival is arbitrary, is moral judgment arbitrary too?"
    )

    print(f"\nPROMPT:\n{prompt}")
    print("-"*80)
    print("Running consensus process (may require multiple iterations)...")
    print("-"*80)

    result = consensus.invoke(prompt)

    print("\nRESULT:")
    result_md = dedent(f"""
        ## Consensus Reached: {result['consensus']}

        ## Final Answer
        {result['final_answer']}

        ## Notes
        {result['notes']}
    """)
    console.print(Markdown(result_md))
    print("="*80)


def test_web_search_no_schema():
    """
    Test Consensus with web search and no structured output.

    This test uses web search to get current information and returns
    the full agent result without a custom schema.
    """
    print("\n" + "="*80)
    print("CONSENSUS TEST - Web Search Without Structured Output")
    print("="*80)

    consensus = Consensus(
        models=[
            "openai:gpt-5-mini",
            "google_genai:gemini-3-flash-preview",
            "anthropic:claude-3-5-haiku-20241022",
            "xai:grok-3-mini",
        ]
    )

    prompt = (
        "What are the latest developments in quantum computing?\n\n"
        "Use the web search to research current news and breakthroughs."
    )

    print(f"\nPROMPT:\n{prompt}")
    print("-"*80)
    print("Running consensus process (may require multiple iterations)...")
    print("-"*80)

    result = consensus.invoke(prompt)

    print("\nRESULT:")
    console.print(Markdown(result['messages'][-1].content))
    print("="*80)


if __name__ == "__main__":
    print("Running Consensus Tests...")

    test_schema()
    test_web_search_no_schema()

    print("\n✅ Tests completed successfully!")
