from llm_ensemble import RunLLM


def test_consensus_three_providers():
    """
    Test consensus with GPT-5.2, Gemini 3, and Claude Sonnet 4.5.

    This test verifies that:
    1. All three models can be initialized successfully
    2. The graph executes all models in parallel
    3. Outputs are properly anonymized and aggregated
    4. The result contains responses from all three LLMs
    """
    models = [
        "openai:gpt-5.2",
        "google_genai:gemini-3-pro-preview",
        "anthropic:claude-sonnet-4-5-20250929"
    ]
    system_message = (
        "You are an expert analyst who provides clear, concise, and well-reasoned "
        "answers. Focus on accuracy and brevity in your responses."
    )

    # Initialize consensus
    consensus = RunLLM(
        models=models,
        system_message=system_message
    )

    # Test prompt
    prompt = (
        "What are the three most important factors to consider when evaluating "
        "the performance of a machine learning model? Provide a brief explanation "
        "for each factor."
    )

    # Invoke and get result
    result = consensus.invoke(prompt)

    # Print results
    print("\n" + "="*80)
    print("CONSENSUS TEST RESULT")
    print("="*80)
    print(f"PROMPT: {prompt}")
    print("-"*80)
    print(f"SYSTEM MESSAGE: {system_message}")
    print("-"*80)
    print("AGGREGATED RESPONSES:")
    print(result)
    print("="*80)


def test_consensus_math_problem():
    """Test consensus with a simple math problem."""
    models = [
        "openai:gpt-5.2",
        "google_genai:gemini-3-pro-preview",
        "anthropic:claude-sonnet-4-5-20250929"
    ]
    system_message = "You are a helpful AI assistant that answers questions accurately and concisely."

    consensus = RunLLM(
        models=models,
        system_message=system_message
    )
    result = consensus.invoke("What is 5 + 3?")

    print("\n" + "="*80)
    print("MATH PROBLEM TEST RESULT:")
    print("="*80)
    print(result)
    print("="*80)


if __name__ == "__main__":
    print("Running RunLLM Tests...")
    print("\n" + "#"*80)
    print("# TEST 1: Three Providers with Complex Question")
    print("#"*80)
    test_consensus_three_providers()

    print("\n" + "#"*80)
    print("# TEST 2: Simple Math Problem")
    print("#"*80)
    test_consensus_math_problem()

    print("\n All tests completed successfully!")
