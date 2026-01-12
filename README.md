# LLM Ensemble

![LLM Ensemble Banner](.github/banner.png)

A Python library for achieving consensus across multiple Large Language Models (LLMs).

## Features

- **Consensus**: Uses a moderator to iteratively coordinate multiple LLMs until consensus is reached
- Can use any model supported by Langchain 
- Supports web search for real-time data 

## Installation

```bash
# Clone from GitHub
git clone https://github.com/zzzrbx/llm-ensemble.git
cd llm-ensemble

# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

## Environment Setup

Create a `.env` file with your API keys, for example:

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

**Judge Model:** Configurable (default: `anthropic:claude-opus-4-5-20251101`). It is recommended to use a high-reasoning model.

**Tools currently available for LLMs:**
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
- **Dynamic queries** - Judge crafts different prompts each iteration:
  - Iteration 1: Sends initial question with research instructions
  - Iteration 2+: Summarizes agreements, highlights disagreements, requests refinements
  - Final iteration: Presents refined consensus statement for confirmation
- **Error handling** - Returns default values on timeout or tool call limit reached

## Examples

### Example 1: With custom schema

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

### Example 2: No custom schema and no web search 

```python
from llm_ensemble import Consensus

# No response_schema - returns full agent result
consensus = Consensus()

result = consensus.invoke(
    "What are the key differences between deontological and consequentialist ethics?"
)

# Access full agent result
print(result['messages'][-1].content)
```

### Example 3: Explicitly setting models

```python
from typing import TypedDict
from llm_ensemble import Consensus

# Explicitly set which models to use for consensus
consensus = Consensus(
    models=[
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-20241022",
        "google:gemini-2.0-flash-exp"
    ],
    judge_model="anthropic:claude-opus-4-5-20251101",
)

result = consensus.invoke(
    "What is the capital of France?\n\n"
)

print(f"Consensus: {result['consensus']}")
print(f"Answer: {result['final_answer']}")
```

## Debugging and Observability

The library integrates with LangSmith for trace observability. Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in your `.env` file to enable tracing.

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
