"""
Example usage of the Tool Call Agent with Guardrails.

This script demonstrates the three main guardrail scenarios:
1. Max tool calls per turn (3 calls limit)
2. Token rate limiting
3. Human review escalation
"""

from agent.graph import (
    graph,
    GuardrailMetrics,
    DEFAULT_GUARDRAILS,
)
from langchain_core.messages import HumanMessage


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def example_1_normal_operation():
    """Example 1: Normal operation - agent makes 1-2 tool calls."""
    print("🟢 EXAMPLE 1: Normal Operation (1-2 tool calls)")
    print_separator()

    initial_state = {
        "messages": [HumanMessage(content="What's the current time?")],
        "guardrail_metrics": GuardrailMetrics().to_dict(),
        "config": DEFAULT_GUARDRAILS,
        "needs_human_review": False,
        "human_review_reason": "",
    }

    result = graph.invoke(initial_state)

    print(f"✅ Completed successfully!")
    print(f"📊 Metrics:")
    print(f"   - Tool calls this turn: {result['guardrail_metrics']['tool_calls_this_turn']}")
    print(f"   - Total tool calls: {result['guardrail_metrics']['total_tool_calls']}")
    print(f"   - Violations: {result['guardrail_metrics']['violations']}")
    print_separator()


def example_2_tool_call_limit():
    """Example 2: Trigger tool call limit (>3 calls)."""
    print("🟡 EXAMPLE 2: Tool Call Limit Exceeded (Simulated)")
    print_separator()

    # Pre-populate metrics to simulate 3 previous tool calls
    metrics = GuardrailMetrics()
    metrics.tool_calls_this_turn = 3  # Already at limit

    initial_state = {
        "messages": [
            HumanMessage(content="Search for multiple things and calculate results")
        ],
        "guardrail_metrics": metrics.to_dict(),
        "config": DEFAULT_GUARDRAILS,
        "needs_human_review": False,
        "human_review_reason": "",
    }

    result = graph.invoke(initial_state)

    print(f"⚠️  Guardrail triggered!")
    print(f"📊 Metrics:")
    print(f"   - Tool calls this turn: {result['guardrail_metrics']['tool_calls_this_turn']}")
    print(f"   - Violations: {result['guardrail_metrics']['violations']}")
    print(f"   - Human review needed: {result.get('needs_human_review', False)}")
    print_separator()


def example_3_token_limit():
    """Example 3: Trigger token rate limit."""
    print("🔴 EXAMPLE 3: Token Rate Limit Exceeded")
    print_separator()

    # Pre-populate metrics to simulate high token usage
    metrics = GuardrailMetrics()
    metrics.tokens_last_minute = 16000  # Over the 15000 limit

    initial_state = {
        "messages": [HumanMessage(content="Quick question about something simple")],
        "guardrail_metrics": metrics.to_dict(),
        "config": DEFAULT_GUARDRAILS,
        "needs_human_review": False,
        "human_review_reason": "",
    }

    result = graph.invoke(initial_state)

    print(f"⚠️  Guardrail triggered!")
    print(f"📊 Metrics:")
    print(f"   - Tokens last minute: {result['guardrail_metrics']['tokens_last_minute']}")
    print(f"   - Violations: {result['guardrail_metrics']['violations']}")
    print(f"   - Human review needed: {result.get('needs_human_review', False)}")
    print_separator()


def example_4_custom_config():
    """Example 4: Custom guardrail configuration."""
    print("⚙️  EXAMPLE 4: Custom Guardrail Configuration")
    print_separator()

    custom_config = {
        "max_tool_calls_per_turn": 5,  # Increased limit
        "max_tokens_per_minute": 20000,  # Increased limit
        "enable_exponential_backoff": True,
        "base_retry_delay": 0.5,
        "max_retry_delay": 10.0,
        "enable_human_review": False,  # Disabled for this example
    }

    initial_state = {
        "messages": [HumanMessage(content="Complex multi-step query")],
        "guardrail_metrics": GuardrailMetrics().to_dict(),
        "config": custom_config,
        "needs_human_review": False,
        "human_review_reason": "",
    }

    result = graph.invoke(initial_state)

    print(f"✅ Completed with custom config!")
    print(f"📊 Custom Config:")
    print(f"   - Max tool calls: {custom_config['max_tool_calls_per_turn']}")
    print(f"   - Max tokens/min: {custom_config['max_tokens_per_minute']}")
    print(f"   - Human review: {custom_config['enable_human_review']}")
    print_separator()


def main():
    """Run all examples."""
    print("\n" + "🚀 Tool Call Agent - Guardrail Examples" + "\n")
    print("This demonstrates the guardrail system preventing infinite loops")
    print("and excessive resource usage in tool-calling agents.")
    print_separator()

    try:
        # Run examples
        example_1_normal_operation()
        input("Press Enter to continue to Example 2...")

        example_2_tool_call_limit()
        input("Press Enter to continue to Example 3...")

        example_3_token_limit()
        input("Press Enter to continue to Example 4...")

        example_4_custom_config()

        print("\n✨ All examples completed!")
        print("\nKey Takeaways:")
        print("  1. Hard cap of 3 tool calls prevents infinite loops")
        print("  2. Token limits prevent cost overruns")
        print("  3. Human review for violation cases")
        print("  4. Configurable guardrails for different use cases")
        print("  5. Trace IDs enable debugging of specific failures")

    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
