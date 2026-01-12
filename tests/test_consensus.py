import sys
from pathlib import Path
from typing import TypedDict

# Add parent directory to path to import llm-ensemble module
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_ensemble import Consensus


class UserSchema(TypedDict):
    """Output schema for sustainability consensus test."""
    consensus: bool  # Whether consensus was reached among LLMs
    final_answer: str  # The agreed-upon answer to the question
    notes: str  # Process insights, key points of agreement/disagreement


def test_sustainability_disclosure():
    """
    Test Consensus with a complex sustainability governance question.

    This test explores whether mandatory sustainability disclosure creates
    genuine accountability or merely shifts responsibility from governments
    to corporations.
    """
    print("\n" + "="*80)
    print("CONSENSUS TEST - Sustainability Disclosure & Accountability")
    print("="*80)

    # Initialize Consensus with custom schema
    consensus = Consensus(response_schema=UserSchema)

    # Complex question about civil disobedience and justice
    prompt = (
        "If survival is arbitrary, is moral judgment arbitrary too?"
    )

    print(f"\nPROMPT:\n{prompt}")
    print("-"*80)
    print("Running consensus process (may require multiple iterations)...")
    print("-"*80)

    # Invoke consensus process
    result = consensus.invoke(prompt)

    # Print results
    print("\nRESULT:")
    print(f"Consensus Reached: {result['consensus']}")
    print(f"\nFinal Answer:\n{result['final_answer']}")
    print(f"\nNotes:\n{result['notes']}")
    print("="*80)


if __name__ == "__main__":
    print("Running Sustainability Consensus Test...")

    test_sustainability_disclosure()

    print("\n✅ Test completed successfully!")
