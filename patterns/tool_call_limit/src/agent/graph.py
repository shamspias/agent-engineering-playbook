"""
Implements:
- Hard cap: max 3 tool calls per turn
- Exponential backoff + jitter for retries
- Cost/token tracking and limits
- Trace IDs for debugging
- Human-in-the-loop escalation
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


# Configure structured logging with custom formatter
class TraceIdFilter(logging.Filter):
    """Add trace_id to log records if not present."""

    def filter(self, record):
        if not hasattr(record, "trace_id"):
            record.trace_id = "N/A"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] - %(message)s",
)
logger = logging.getLogger(__name__)
logger.addFilter(TraceIdFilter())


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
class GuardrailConfig(TypedDict, total=False):
    """Configurable guardrail parameters."""

    max_tool_calls_per_turn: int  # Default: 3
    max_tokens_per_minute: int  # Default: 15000
    enable_exponential_backoff: bool  # Default: True
    base_retry_delay: float  # Default: 1.0 (seconds)
    max_retry_delay: float  # Default: 30.0 (seconds)
    enable_human_review: bool  # Default: True


DEFAULT_GUARDRAILS: GuardrailConfig = {
    "max_tool_calls_per_turn": 3,
    "max_tokens_per_minute": 15000,
    "enable_exponential_backoff": True,
    "base_retry_delay": 1.0,
    "max_retry_delay": 30.0,
    "enable_human_review": True,
}


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
@dataclass
class GuardrailMetrics:
    """Track guardrail metrics across the conversation."""

    tool_calls_this_turn: int = 0
    total_tool_calls: int = 0
    tokens_last_minute: int = 0
    retry_count: int = 0
    last_token_reset: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardrailMetrics":
        """Create from dict for deserialization."""
        return cls(**data)


class AgentState(TypedDict):
    """State for the agent with guardrail tracking."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    guardrail_metrics: dict[str, Any]  # Changed to dict for LangGraph compatibility
    config: GuardrailConfig
    needs_human_review: bool
    human_review_reason: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_metrics(state: AgentState) -> GuardrailMetrics:
    """Extract metrics from state, handling both dict and object formats."""
    metrics_data = state["guardrail_metrics"]
    if isinstance(metrics_data, dict):
        return GuardrailMetrics.from_dict(metrics_data)
    return metrics_data


def update_metrics(state: AgentState, metrics: GuardrailMetrics) -> dict[str, Any]:
    """Update state with new metrics."""
    return {"guardrail_metrics": metrics.to_dict()}


# ============================================================================
# TOOLS
# ============================================================================
@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results as a string
    """
    logger.info(f"Executing search_web with query: {query}")
    # Simulate API call
    time.sleep(0.1)
    return f"Search results for '{query}': [Mock results - implement real search here]"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    logger.info(f"Executing calculate with expression: {expression}")
    try:
        # Safe eval for demo purposes - use a proper math parser in production
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current time.

    Returns:
        Current timestamp
    """
    logger.info("Executing get_current_time")
    return f"Current time: {datetime.now().isoformat()}"


# Tool registry
TOOLS = [search_web, calculate, get_current_time]
tool_node = ToolNode(TOOLS)


# ============================================================================
# GUARDRAIL FUNCTIONS
# ============================================================================
def calculate_backoff_delay(
    retry_count: int, base_delay: float, max_delay: float
) -> float:
    """Calculate exponential backoff with jitter.

    Args:
        retry_count: Number of retries so far
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds with jitter applied
    """
    exponential_delay = base_delay * (2**retry_count)
    capped_delay = min(exponential_delay, max_delay)
    jitter = random.uniform(0, capped_delay * 0.1)  # 10% jitter
    return capped_delay + jitter


def check_guardrails(state: AgentState) -> dict[str, Any]:
    """Check if any guardrails have been violated.

    Args:
        state: Current agent state

    Returns:
        Updated state with guardrail status
    """
    metrics = get_metrics(state)
    config = state.get("config", DEFAULT_GUARDRAILS)

    trace_id = metrics.trace_id
    violations = []

    # Check tool call limit
    if metrics.tool_calls_this_turn >= config["max_tool_calls_per_turn"]:
        violation = f"Tool call limit exceeded: {metrics.tool_calls_this_turn}/{config['max_tool_calls_per_turn']}"
        violations.append(violation)
        logger.warning(violation, extra={"trace_id": trace_id})

    # Check token rate limit
    current_time = time.time()
    time_since_reset = current_time - metrics.last_token_reset

    if time_since_reset >= 60.0:  # Reset every minute
        metrics.tokens_last_minute = 0
        metrics.last_token_reset = current_time
    elif metrics.tokens_last_minute > config["max_tokens_per_minute"]:
        violation = f"Token rate limit exceeded: {metrics.tokens_last_minute}/{config['max_tokens_per_minute']} tokens/min"
        violations.append(violation)
        logger.warning(violation, extra={"trace_id": trace_id})

    # Update metrics
    metrics.violations = violations
    needs_review = len(violations) > 0 and config.get("enable_human_review", True)

    return {
        "guardrail_metrics": metrics.to_dict(),
        "needs_human_review": needs_review,
        "human_review_reason": "; ".join(violations) if violations else "",
    }


# ============================================================================
# GRAPH NODES
# ============================================================================
def call_model(state: AgentState) -> dict[str, Any]:
    """Call the language model (simulated).

    In production, replace with actual LLM integration.
    """
    metrics = get_metrics(state)
    messages = state["messages"]
    config = state.get("config", DEFAULT_GUARDRAILS)

    logger.info(
        f"Calling model with {len(messages)} messages",
        extra={"trace_id": metrics.trace_id},
    )

    # Simulate LLM response with tool calls
    # In production, replace with actual LLM call
    last_message = messages[-1]

    # Simulate token usage
    estimated_tokens = len(str(last_message.content)) * 2
    metrics.tokens_last_minute += estimated_tokens

    # Determine if we should make more tool calls or provide a final answer
    # In a real LLM, this would be based on the model's reasoning
    should_call_tools = _should_generate_tool_calls(
        messages, metrics, config
    )

    if should_call_tools:
        # Simulate tool calling - can generate 1-2 tool calls per turn
        num_tools_to_call = min(
            random.randint(1, 2),
            config["max_tool_calls_per_turn"] - metrics.tool_calls_this_turn
        )

        tool_calls = []
        for i in range(num_tools_to_call):
            # Randomly pick a tool to simulate realistic behavior
            tool_name = random.choice(["search_web", "calculate", "get_current_time"])
            tool_calls.append({
                "name": tool_name,
                "args": _generate_tool_args(tool_name, i),
                "id": f"call_{uuid.uuid4().hex[:8]}",
            })

        response = AIMessage(
            content="",
            tool_calls=tool_calls,
        )
    else:
        # Provide final answer
        response = AIMessage(
            content="Based on the tool results, here's my answer: [Final answer would go here]"
        )

    logger.info(
        f"Model response generated. Has tool calls: {bool(getattr(response, 'tool_calls', None))}, "
        f"Num tool calls: {len(getattr(response, 'tool_calls', []))}",
        extra={"trace_id": metrics.trace_id},
    )

    return {
        "messages": [response],
        "guardrail_metrics": metrics.to_dict(),
    }


def _should_generate_tool_calls(
    messages: Sequence[BaseMessage],
    metrics: GuardrailMetrics,
    config: GuardrailConfig
) -> bool:
    """Determine if the model should generate tool calls.

    This simulates the LLM's decision-making process. In production,
    the actual LLM would make this decision based on the conversation context.

    The agent tries to make multiple calls to gather information, but stops
    naturally after a reasonable amount. The guardrails kick in if the agent
    tries to exceed the configured limits.

    Args:
        messages: Conversation history
        metrics: Current guardrail metrics
        config: Guardrail configuration

    Returns:
        True if should generate tool calls, False if should provide final answer
    """
    # Don't make tool calls if we've hit or exceeded the limit
    if metrics.tool_calls_this_turn >= config["max_tool_calls_per_turn"]:
        return False

    # Check if we have any tool messages (results from previous tool calls)
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)

    # Simulate realistic agent behavior:
    # - The agent wants to make 2-3 tool calls typically
    # - It stops naturally after gathering sufficient information
    # - Guardrails prevent it if it tries to exceed the limit

    if not has_tool_results:
        # First turn - always gather information
        return True
    elif metrics.tool_calls_this_turn < 2:
        # Made 1 call - usually wants more information
        # 80% chance to continue
        return random.random() < 0.8
    elif metrics.tool_calls_this_turn < 3:
        # Made 2 calls - sometimes wants one more
        # 40% chance to continue
        return random.random() < 0.4
    else:
        # Made 3+ calls - usually has enough information
        # Rarely continues (10% chance)
        # This is where guardrails would catch runaway agents
        return random.random() < 0.1


def _generate_tool_args(tool_name: str, call_index: int) -> dict[str, str]:
    """Generate appropriate arguments for each tool type.

    Args:
        tool_name: Name of the tool
        call_index: Index of this call in the current batch

    Returns:
        Dictionary of tool arguments
    """
    if tool_name == "search_web":
        queries = [
            "example query",
            "related information",
            "additional context",
            "verification search"
        ]
        return {"query": queries[call_index % len(queries)]}
    elif tool_name == "calculate":
        expressions = ["2 + 2", "10 * 5", "100 / 4", "7 ** 2"]
        return {"expression": expressions[call_index % len(expressions)]}
    elif tool_name == "get_current_time":
        return {}
    else:
        return {}


def execute_tools(state: AgentState) -> dict[str, Any]:
    """Execute tool calls with retry logic and backoff.

    Args:
        state: Current agent state

    Returns:
        Updated state with tool results
    """
    metrics = get_metrics(state)
    config = state.get("config", DEFAULT_GUARDRAILS)
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    # Increment tool call counter
    num_calls = len(last_message.tool_calls)
    metrics.tool_calls_this_turn += num_calls
    metrics.total_tool_calls += num_calls

    logger.info(
        f"Executing {num_calls} tool calls. Turn count: {metrics.tool_calls_this_turn}",
        extra={"trace_id": metrics.trace_id},
    )

    # Apply exponential backoff if enabled and we're retrying
    if config.get("enable_exponential_backoff", True) and metrics.retry_count > 0:
        delay = calculate_backoff_delay(
            metrics.retry_count,
            config.get("base_retry_delay", 1.0),
            config.get("max_retry_delay", 30.0),
        )
        logger.info(
            f"Applying backoff delay: {delay:.2f}s (retry #{metrics.retry_count})",
            extra={"trace_id": metrics.trace_id},
        )
        time.sleep(delay)

    # Execute tools using the ToolNode
    result = tool_node.invoke(state)

    # Reset retry count on successful execution
    metrics.retry_count = 0

    return {
        "messages": result["messages"],
        "guardrail_metrics": metrics.to_dict(),
    }


def human_review(state: AgentState) -> dict[str, Any]:
    """Handle human-in-the-loop review when guardrails are violated.

    Args:
        state: Current agent state

    Returns:
        State with human review message
    """
    metrics = get_metrics(state)
    reason = state.get("human_review_reason", "Unknown violation")

    logger.warning(
        f"Escalating to human review. Reason: {reason}",
        extra={"trace_id": metrics.trace_id},
    )

    # Create a message for human review
    review_message = HumanMessage(
        content=f"""
🔴 HUMAN REVIEW REQUIRED 🔴

Trace ID: {metrics.trace_id}
Reason: {reason}

Metrics:
- Tool calls this turn: {metrics.tool_calls_this_turn}
- Total tool calls: {metrics.total_tool_calls}
- Tokens last minute: {metrics.tokens_last_minute}
- Retry count: {metrics.retry_count}

Please review the conversation and decide how to proceed.
Options: approve, reject, or modify the request.
"""
    )

    return {
        "messages": [review_message],
        "needs_human_review": False,  # Reset flag
    }


# ============================================================================
# ROUTING LOGIC
# ============================================================================
def should_continue(
    state: AgentState,
) -> Literal["execute_tools", "end"]:
    """Determine next step based on the last message.

    Args:
        state: Current agent state

    Returns:
        Next node to execute
    """
    messages = state["messages"]
    last_message = messages[-1]
    metrics = get_metrics(state)

    logger.debug(
        f"Routing decision. Last message type: {type(last_message).__name__}, "
        f"Tool calls this turn: {metrics.tool_calls_this_turn}",
        extra={"trace_id": metrics.trace_id},
    )

    # If last message has tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.debug(
            f"Routing to execute_tools. Number of tool calls: {len(last_message.tool_calls)}",
            extra={"trace_id": metrics.trace_id},
        )
        return "execute_tools"

    # Otherwise, we're done
    logger.debug(
        "No tool calls detected. Routing to end.",
        extra={"trace_id": metrics.trace_id},
    )
    return "end"


def check_guardrails_routing(
    state: AgentState,
) -> Literal["human_review", "call_model"]:
    """Route to human review if guardrails violated, otherwise continue.

    Args:
        state: Current agent state

    Returns:
        Next node to execute
    """
    if state.get("needs_human_review", False):
        return "human_review"
    return "call_model"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
def create_agent_graph() -> StateGraph:
    """Create the agent graph with guardrails.

    Returns:
        Compiled StateGraph
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("check_guardrails", check_guardrails)
    workflow.add_node("human_review", human_review)

    # Set entry point
    workflow.add_edge(START, "call_model")

    # Add conditional routing after model call
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )

    # After tool execution, check guardrails
    workflow.add_edge("execute_tools", "check_guardrails")

    # After guardrail check, either go to human review or continue
    workflow.add_conditional_edges(
        "check_guardrails",
        check_guardrails_routing,
        {
            "human_review": "human_review",
            "call_model": "call_model",
        },
    )

    # After human review, end the conversation
    workflow.add_edge("human_review", END)

    return workflow.compile()


# Create the compiled graph
graph = create_agent_graph()