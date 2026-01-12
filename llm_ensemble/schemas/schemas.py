from typing import TypedDict, Annotated
from operator import or_


class InputState(TypedDict):
    """Input schema: what users provide when invoking."""
    prompt: str


class OutputState(TypedDict):
    """Output schema: what the graph returns."""
    result: str


class RunLLMState(TypedDict):
    """Overall state: internal state during execution."""
    prompt: str
    # Use or_ reducer to merge dict updates from parallel nodes
    model_outputs: Annotated[dict[str, str], or_]
    result: str
