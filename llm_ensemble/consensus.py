from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware, ToolCallLimitMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from .run_llm import RunLLM
from pathlib import Path
from typing import Type


class Consensus:
    """
    Consensus class that uses a configurable judge model to orchestrate
    multiple RunLLM invocations until consensus is reached among LLMs.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        judge_model: str = "anthropic:claude-opus-4-5-20251101",
        summarization_trigger_tokens: int = 200000,
        summarization_keep_messages: int = 5,
        run_limit: int = 20,
        response_schema: Type | None = None
    ):
        """
        Initialize the Consensus class.

        Args:
            models: List of model strings in format "provider:model-name".
                   Defaults to GPT-5-mini, Gemini 3 Flash, Claude 3.5 Haiku, Grok 3 Mini.
            judge_model: Model string for the judge coordinator in format "provider:model-name".
                        Defaults to "anthropic:claude-opus-4-5-20251101".
            summarization_trigger_tokens: Token count to trigger summarization middleware.
            summarization_keep_messages: Number of messages to keep after summarization.
            run_limit: Maximum number of calls to run_llms tool per invocation.
            response_schema: Optional schema for structured output (TypedDict or Pydantic model).
                            If None, returns full agent result without structured output.
        """
        # Set default models
        if models is None:
            models = [
                "openai:gpt-5-mini",
                "google_genai:gemini-3-flash-preview",
                "anthropic:claude-3-5-haiku-20241022",
                "xai:grok-3-mini",
        ]

        # Load judge prompt
        judge_prompt_path = Path(__file__).parent / "prompts" / "judge.prompt"
        judge_prompt = judge_prompt_path.read_text()

        # Set basic system message for RunLLM
        system_message = "You are a helpful AI assistant."

        # Create tool function as closure (captures models and system_message)
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
            run_llm = RunLLM(models=models, system_message=system_message)
            return run_llm.invoke(query)

        # Create judge LLM using init_chat_model
        llm = init_chat_model(judge_model)

        # Create middleware
        middleware = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=lambda rt: StateBackend(rt)),
            SummarizationMiddleware(
                model="anthropic:claude-3-5-sonnet-20241022",
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

    def invoke(self, prompt: str):
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
