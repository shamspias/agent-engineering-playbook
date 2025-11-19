"""Unit tests for configuration and guardrail logic."""

import pytest
from langgraph.pregel import Pregel

from agent.graph import (
    DEFAULT_GUARDRAILS,
    GuardrailMetrics,
    calculate_backoff_delay,
    graph,
)


def test_graph_is_compiled():
    """Test that the graph is properly compiled."""
    assert isinstance(graph, Pregel)


def test_default_guardrails():
    """Test default guardrail configuration."""
    assert DEFAULT_GUARDRAILS["max_tool_calls_per_turn"] == 3
    assert DEFAULT_GUARDRAILS["max_tokens_per_minute"] == 15000
    assert DEFAULT_GUARDRAILS["enable_exponential_backoff"] is True
    assert DEFAULT_GUARDRAILS["enable_human_review"] is True


def test_guardrail_metrics_initialization():
    """Test GuardrailMetrics dataclass initialization."""
    metrics = GuardrailMetrics()

    assert metrics.tool_calls_this_turn == 0
    assert metrics.total_tool_calls == 0
    assert metrics.tokens_last_minute == 0
    assert metrics.retry_count == 0
    assert metrics.violations == []
    assert metrics.trace_id is not None
    assert len(metrics.trace_id) > 0


def test_guardrail_metrics_to_dict():
    """Test metrics serialization to dict."""
    metrics = GuardrailMetrics(
        tool_calls_this_turn=2,
        total_tool_calls=5,
        tokens_last_minute=1000,
    )

    metrics_dict = metrics.to_dict()

    assert isinstance(metrics_dict, dict)
    assert metrics_dict["tool_calls_this_turn"] == 2
    assert metrics_dict["total_tool_calls"] == 5
    assert metrics_dict["tokens_last_minute"] == 1000


def test_guardrail_metrics_from_dict():
    """Test metrics deserialization from dict."""
    data = {
        "tool_calls_this_turn": 3,
        "total_tool_calls": 10,
        "tokens_last_minute": 5000,
        "retry_count": 1,
        "last_token_reset": 0,
        "trace_id": "test-trace-id",
        "violations": ["test violation"],
    }

    metrics = GuardrailMetrics.from_dict(data)

    assert metrics.tool_calls_this_turn == 3
    assert metrics.total_tool_calls == 10
    assert metrics.tokens_last_minute == 5000
    assert metrics.retry_count == 1
    assert metrics.trace_id == "test-trace-id"
    assert len(metrics.violations) == 1


def test_exponential_backoff_calculation():
    """Test exponential backoff delay calculation."""
    # First retry: ~1s
    delay_1 = calculate_backoff_delay(retry_count=0, base_delay=1.0, max_delay=30.0)
    assert 1.0 <= delay_1 <= 1.1  # Base + 10% jitter

    # Second retry: ~2s
    delay_2 = calculate_backoff_delay(retry_count=1, base_delay=1.0, max_delay=30.0)
    assert 2.0 <= delay_2 <= 2.2  # 2^1 * base + jitter

    # Third retry: ~4s
    delay_3 = calculate_backoff_delay(retry_count=2, base_delay=1.0, max_delay=30.0)
    assert 4.0 <= delay_3 <= 4.4  # 2^2 * base + jitter


def test_exponential_backoff_max_cap():
    """Test that backoff delay is capped at max_delay."""
    # Very high retry count should be capped
    delay = calculate_backoff_delay(retry_count=10, base_delay=1.0, max_delay=10.0)
    assert delay <= 11.0  # Max delay + max jitter (10%)


def test_exponential_backoff_custom_base():
    """Test backoff with custom base delay."""
    delay = calculate_backoff_delay(retry_count=0, base_delay=2.0, max_delay=30.0)
    assert 2.0 <= delay <= 2.2  # Custom base + jitter


class TestGuardrailMetrics:
    """Test suite for GuardrailMetrics."""

    def test_metrics_increment(self):
        """Test incrementing metrics."""
        metrics = GuardrailMetrics()

        metrics.tool_calls_this_turn += 1
        metrics.total_tool_calls += 1

        assert metrics.tool_calls_this_turn == 1
        assert metrics.total_tool_calls == 1

    def test_metrics_reset_turn(self):
        """Test resetting per-turn metrics."""
        metrics = GuardrailMetrics()
        metrics.tool_calls_this_turn = 3
        metrics.total_tool_calls = 10

        # Reset turn metrics
        metrics.tool_calls_this_turn = 0

        assert metrics.tool_calls_this_turn == 0
        assert metrics.total_tool_calls == 10  # Shouldn't reset

    def test_violations_list(self):
        """Test violations list management."""
        metrics = GuardrailMetrics()

        assert len(metrics.violations) == 0

        metrics.violations.append("Tool call limit exceeded")
        assert len(metrics.violations) == 1

        metrics.violations.append("Token limit exceeded")
        assert len(metrics.violations) == 2


class TestConfiguration:
    """Test suite for configuration."""

    def test_custom_config_override(self):
        """Test custom configuration overrides defaults."""
        custom_config = {
            "max_tool_calls_per_turn": 5,
            "max_tokens_per_minute": 20000,
            "enable_human_review": False,
        }

        assert custom_config["max_tool_calls_per_turn"] == 5
        assert custom_config["max_tokens_per_minute"] == 20000
        assert custom_config["enable_human_review"] is False

    def test_config_partial_override(self):
        """Test partial configuration override."""
        config = DEFAULT_GUARDRAILS.copy()
        config["max_tool_calls_per_turn"] = 10

        assert config["max_tool_calls_per_turn"] == 10
        assert config["max_tokens_per_minute"] == 15000  # Still default
