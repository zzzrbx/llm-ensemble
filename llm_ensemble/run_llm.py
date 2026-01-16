from typing import Any, Callable

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from .utils import add, subtract, multiply, divide, search_the_web
from .schemas import InputState, OutputState, RunLLMState

# Load environment variables at module level
load_dotenv()


class RunLLM:
    """
    RunLLM class that runs multiple LLM agents in parallel on the same query,
    then aggregates their responses.
    """

    def __init__(self, models: list[str], system_message: str) -> None:
        """
        Initialize the RunLLM class.

        Args:
            models: List of model strings in format "provider:model-name"
            system_message: System message to include in every agent invocation
        """
        self._models = models
        self._system_message = system_message

        # Create tool list
        tools = [
            search_the_web,
            add,
            subtract,
            multiply,
            divide
        ]

        # Initialize agents for each model
        self._agents = {}
        for model_string in models:
            # Pass model string directly to create_agent
            # LangChain's init_chat_model handles provider parsing automatically
            agent = create_agent(
                model=model_string,
                tools=tools,
                system_prompt=self._system_message
            )
            self._agents[model_string] = agent

        # Build and compile the StateGraph
        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build and compile the StateGraph.

        Returns:
            Compiled graph ready for invocation
        """
        # Initialize StateGraph with input/output schemas
        graph = StateGraph(
            state_schema=RunLLMState,
            input_schema=InputState,
            output_schema=OutputState
        )

        # Factory function to create model nodes with proper closure
        def make_model_node(model_name: str, agent: Any) -> Callable[[RunLLMState], dict]:
            """Factory function to create a node function with proper closure."""
            def node_function(state: RunLLMState) -> dict:
                prompt = state["prompt"]
                # Invoke agent with user prompt
                result = agent.invoke({
                    "messages": [{"role": "user", "content": prompt}]
                })
                # Extract final AI message content
                content = result["messages"][-1].content

                # Handle different content formats from different providers
                # Gemini returns: [{'type': 'text', 'text': '...', 'extras': {...}}]
                # OpenAI/Anthropic return: plain string
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from Gemini's structured format
                    output_text = content[0].get('text', str(content))
                else:
                    # Use content directly for OpenAI/Anthropic
                    output_text = content

                # Return update to model_outputs dict
                return {"model_outputs": {model_name: output_text}}
            return node_function

        # Add a node for each model
        # Use sanitized node names (replace : with _) since LangGraph doesn't allow colons
        for model_name in self._models:
            node_fn = make_model_node(
                model_name,
                self._agents[model_name]
            )
            # Sanitize node name by replacing : with _
            sanitized_node_name = model_name.replace(":", "_")
            graph.add_node(sanitized_node_name, node_fn)

        # Add process node
        def process_node(state: RunLLMState) -> dict:
            """Aggregate all model outputs in order."""
            outputs = []
            for model_name in self._models:
                output = state["model_outputs"].get(model_name, "")
                outputs.append(f"{model_name}:\n{output}")

            aggregated = "\n\n".join(outputs)
            return {"result": aggregated}

        graph.add_node("process", process_node)

        # Add edges
        # From START to each model node
        for model_name in self._models:
            sanitized_node_name = model_name.replace(":", "_")
            graph.add_edge(START, sanitized_node_name)

        # From each model node to process
        for model_name in self._models:
            sanitized_node_name = model_name.replace(":", "_")
            graph.add_edge(sanitized_node_name, "process")

        # From process to END
        graph.add_edge("process", END)

        # Compile and return
        return graph.compile()

    def invoke(self, prompt: str) -> str:
        """
        Invoke the graph with a prompt.

        Args:
            prompt: The user's query

        Returns:
            Aggregated responses from all models
        """
        # Thanks to input/output schema configuration,
        # we can pass just the prompt string
        result = self._graph.invoke({"prompt": prompt})
        # Result dict contains only the output state fields
        return result["result"]
