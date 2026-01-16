from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware, ToolCallLimitMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from .run_llm import RunLLM
from pathlib import Path
from typing import Any, Type


class Consensus:
    """
    Consensus class that uses a configurable judge model to orchestrate
    multiple RunLLM invocations until consensus is reached among LLMs.
    """

    def __init__(
        self,
        models: list[str],
        judge_model: str = "anthropic:claude-opus-4-5-20251101",
        summarization_model: str = "claude-4-5-sonnet-20250929",
        summarization_trigger_tokens: int = 200_000,
        summarization_keep_messages: int = 5,
        run_limit: int = 20,
        response_schema: Type | None = None
    ) -> None:
        """
        Initialize the Consensus class.

        Args:
            models: List of model strings in format "provider:model-name".
            judge_model: Model string for the judge coordinator in format "provider:model-name".
                        Defaults to "anthropic:claude-opus-4-5-20251101".
            summarization_model: Model string for summarization middleware in format "provider:model-name".
                        Defaults to "anthropic:claude-3-5-sonnet-20241022".
            summarization_trigger_tokens: Token count to trigger summarization middleware.
            summarization_keep_messages: Number of messages to keep after summarization.
            run_limit: Maximum number of calls to run_llms tool per invocation.
            response_schema: Optional schema for structured output (TypedDict or Pydantic model).
                            If None, returns full agent result without structured output.

        Raises:
            ValueError: If models list is empty or contains only one model.
        """
        if not models:
            raise ValueError("models list cannot be empty")
        if len(models) < 2:
            raise ValueError("models list must contain at least 2 models for consensus")

        # Store as instance variables for use in tool creation
        self.models = models
        self.system_message = "You are a helpful AI assistant."

        # Load judge prompt and inject run_limit
        judge_prompt_path = Path(__file__).parent / "prompts" / "judge.prompt"
        judge_prompt = judge_prompt_path.read_text().format(run_limit=run_limit)

        # Create the run_llms tool
        run_llms = self._create_run_llms_tool()

        # Create judge LLM using init_chat_model
        llm = init_chat_model(judge_model)

        # Create middleware
        middleware = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=lambda rt: StateBackend(rt)),
            SummarizationMiddleware(
                model=summarization_model,
                trigger=("tokens", summarization_trigger_tokens),
                keep=("messages", summarization_keep_messages)
            ),
            ToolCallLimitMiddleware(
                tool_name="run_llms",
                run_limit=run_limit,
                exit_behavior="error"
            )
        ]

        # Create agent with optional structured output
        if response_schema is not None:
            self._agent = create_agent(
                model=llm,
                tools=[run_llms],
                system_prompt=judge_prompt,
                middleware=middleware,
                response_format=response_schema
            )
        else:
            self._agent = create_agent(
                model=llm,
                tools=[run_llms],
                system_prompt=judge_prompt,
                middleware=middleware
            )

        self._response_schema = response_schema

    def _create_run_llms_tool(self) -> Any:
        """
        Creates the run_llms tool with access to instance variables.

        Returns:
            A LangChain tool that runs multiple LLMs in parallel.
        """
        @tool
        def run_llms(query: str) -> str:
            """
            Runs multiple LLMs in parallel on the same query.

            Args:
                query: The prompt/question to send to all LLMs. Include full context and
                       any specific instructions (e.g., "use search_the_web for web search",
                       "use add/multiply/subtract/divide tools for calculations").

            Returns:
                Aggregated responses from all LLMs. Each response is prefixed with the
                exact model identifier (e.g., "openai:gpt-5-mini:", "google_genai:gemini-3-flash-preview:").
                Always refer to models by these exact identifiers in your analysis.
            """
            run_llm = RunLLM(models=self.models, system_message=self.system_message)
            return run_llm.invoke(query)

        return run_llms

    def invoke(self, prompt: str) -> dict | None:
        """
        Invoke the consensus process with a user query.

        Args:
            prompt: The user's initial query

        Returns:
            If response_schema was provided: structured response dict (or default values on error)
            Otherwise: full agent result (or None on error)
        """
        try:
            result = self._agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            if self._response_schema is not None:
                return result["structured_response"]
            else:
                return result
        except Exception as e:
            # Tool call limit reached or other error
            print(f"Error during consensus: {str(e)}")
            if self._response_schema is not None:
                # Return default values - create a dict with all schema keys set to defaults
                # This tries to match common schema patterns
                default_dict = {}
                if hasattr(self._response_schema, '__annotations__'):
                    for key, type_hint in self._response_schema.__annotations__.items():
                        if key in ['consensus', 'consensus_reached']:
                            default_dict[key] = False
                        elif type_hint == bool:
                            default_dict[key] = False
                        elif type_hint == str:
                            default_dict[key] = f"Error occurred: {str(e)}"
                        else:
                            default_dict[key] = None
                return default_dict if default_dict else None
            else:
                return None
