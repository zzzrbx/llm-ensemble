# LLM Ensemble

![LLM Ensemble Banner](.github/banner.png)

A Python library for achieving consensus across multiple Large Language Models (LLMs) through an AI judge coordinator.

## Features

- **Consensus**: Uses an AI judge to iteratively coordinate multiple LLMs until consensus is reached
- Support for GPT, Gemini, Claude, and Grok models out of the box
- Built on LangChain and LangGraph for robust agent orchestration
- **Advanced Middleware**: TodoList for task tracking, Filesystem for model name persistence, Summarization for context management, and ToolCallLimit for resource control

## Installation

```bash
# Using uv (recommended)
uv add llm-ensemble

# Using pip
pip install llm-ensemble
```

## Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
XAI_API_KEY=your_xai_key
TAVILY_API_KEY=your_tavily_key  # For web search
```

## Quick Start

Use a deep agent judge to coordinate multiple LLMs until consensus is reached:

```python
from typing import TypedDict
from llm_ensemble import Consensus

# Define your custom output schema
class UserSchema(TypedDict):
    consensus: bool
    final_answer: str
    notes: str

# Initialize with custom schema
consensus = Consensus(response_schema=UserSchema)

# Get consensus on a question
result = consensus.invoke(
    "When is Artificial General Intelligence (AGI) expected to emerge?\n\n"
    "Use the web search to research.\n"
    "- consensus: boolean indicating if consensus was reached\n"
    "- final_answer: the agreed-upon answer to the question\n"
    "- notes: process insights, key agreements or disagreements found"
)

print(f"Consensus: {result['consensus']}")
print(f"\nFinal Answer:\n{result['final_answer']}")
print(f"\nNotes:\n{result['notes']}")
```

## Configuration

```python
Consensus(
    models: list[str] | None = None,                    # Default: GPT-5-mini, Gemini 3 Flash, Claude 3.5 Haiku, Grok 3 Mini
    judge_model: str = "anthropic:claude-opus-4-5-20251101",  # Judge coordinator model (default: Claude Opus 4.5)
    summarization_trigger_tokens: int = 200000,         # Token threshold for summarization
    summarization_keep_messages: int = 5,               # Messages to keep after summarization
    run_limit: int = 20,                                # Max calls to run_llms tool per invocation
    response_schema: Type | None = None                 # Optional TypedDict or Pydantic model for structured output
)
```

**Judge Model:** Configurable (default: `anthropic:claude-opus-4-5-20251101`). Claude Opus 4.5 provides the best reasoning capabilities for consensus coordination.

**Available Tools for LLMs:**
- `search_the_web` - Tavily web search for current events and factual data
- `add`, `subtract`, `multiply`, `divide` - Math operations

**Custom Output Schema:**
You define your own schema based on your needs. Example:
```python
from typing import TypedDict

class MySchema(TypedDict):
    consensus: bool           # Whether consensus was achieved
    final_answer: str         # The agreed-upon answer
    notes: str                # Process insights, agreements, disagreements
```

If `response_schema` is `None`, the full agent result is returned instead of structured output.

## How It Works

```
User Query → Judge (configurable, default: Claude Opus 4.5)
    ↓
Judge calls run_llms tool
    ├── Model A (parallel)
    ├── Model B (parallel)
    ├── Model C (parallel)
    └── Model D (parallel)
         ↓
Judge analyzes responses
    ├── Consensus? → Return answer
    └── No consensus? → Refine query and call run_llms again
         ↓
Repeat until consensus or limit reached
```

**Key Features:**
- Judge is **unbiased** - determines consensus based only on LLM responses, not its own knowledge
- Judge is **model-agnostic** - doesn't favor any LLM based on its name
- **Task tracking** - TodoList middleware helps judge track agreements, disagreements, and next steps
- **Model name persistence** - Filesystem middleware saves exact LLM identifiers to prevent hallucination
- **Dynamic queries** - Judge crafts different prompts each iteration:
  - Iteration 1: Sends initial question with research instructions
  - Iteration 2+: Summarizes agreements, highlights disagreements, requests refinements
  - Final iteration: Presents refined consensus statement for confirmation
- **Context management** - Judge provides full context each time (LLMs are stateless)
- **Tool reminders** - Judge instructs LLMs to use `search_the_web` for current events/factual data, math tools for calculations
- **Error handling** - Returns default values on timeout or tool call limit reached

## Examples

### Example 1: With Custom Schema

```python
from typing import TypedDict
from llm_ensemble import Consensus

class UserSchema(TypedDict):
    consensus: bool
    final_answer: str
    notes: str

consensus = Consensus(response_schema=UserSchema)

result = consensus.invoke(
    "When is Artificial General Intelligence (AGI) expected to emerge?\n\n"
    "Use the web search to research.\n"
    "- consensus: boolean indicating if consensus was reached\n"
    "- final_answer: the agreed-upon answer to the question\n"
    "- notes: process insights, key agreements or disagreements found"
)

print(f"Consensus: {result['consensus']}")
print(f"Answer: {result['final_answer']}")
print(f"Notes: {result['notes']}")
```

### Example 2: Without Structured Output

```python
from llm_ensemble import Consensus

# No response_schema - returns full agent result
consensus = Consensus()

result = consensus.invoke(
    "What are the latest developments in quantum computing as of January 2026?"
)

# Access full agent result
print(result['messages'][-1].content)
```

## Supported Models

### OpenAI
- `openai:gpt-5.2`
- `openai:gpt-4o`
- Any OpenAI model

### Google
- `google_genai:gemini-3-flash-preview`
- `google_genai:gemini-2.0-flash-exp`
- Any Google Gemini model

### Anthropic
- `anthropic:claude-opus-4-5-20251101`
- `anthropic:claude-3-5-haiku-20241022`
- Any Anthropic Claude model

### xAI
- `xai:grok-3-mini`
- `xai:grok-3`
- Any xAI Grok model

## Testing

Run the test suite:

```bash
# Test Consensus with civil disobedience question
uv run python tests/test_consensus.py
```

## Debugging and Observability

The library integrates with LangSmith for trace observability. Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in your `.env` file to enable tracing.

**Analyzing Judge Behavior:**
You can download and analyze traces to see how the judge dynamically refines prompts:

```python
# See llm_ensemble/debug/analyze_judge_prompts.py for an example
# Downloads trace and shows how queries evolved across iterations
```

**What to look for in traces:**
- How many iterations the judge needed to reach consensus
- How the judge summarized agreements and disagreements
- How queries evolved from iteration 1 → 2 → 3
- Which models agreed/disagreed and when

## Development

### Project Structure

```
llm-ensemble/
├── llm_ensemble/
│   ├── __init__.py
│   ├── run_llm.py        # Parallel LLM execution (internal)
│   ├── consensus.py      # Consensus class (main API)
│   ├── utils.py               # Tool definitions
│   ├── schemas/
│   │   └── schemas.py         # State schemas
│   └── prompts/
│       └── judge.prompt       # Judge system prompt
├── tests/
│   └── test_consensus.py      # Consensus tests
└── README.md
```

### Requirements

- Python 3.12+
- LangChain 1.2.3+
- LangGraph 1.0.5+
- DeepAgents 0.1.0+
- API keys for OpenAI, Anthropic, Google, xAI (optional: based on models used)

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with:
- [LangChain](https://langchain.com) - LLM framework
- [LangGraph](https://langchain.com/langgraph) - Agent orchestration
- [DeepAgents](https://github.com/langchain-ai/deepagents) - Deep agent capabilities
