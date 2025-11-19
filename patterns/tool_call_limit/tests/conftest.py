"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio as the async backend."""
    return "asyncio"


@pytest.fixture
def sample_message():
    """Sample message for testing."""
    return {"role": "user", "content": "Test message"}


@pytest.fixture
def default_config():
    """Default guardrail configuration."""
    return {
        "max_tool_calls_per_turn": 3,
        "max_tokens_per_minute": 15000,
        "enable_exponential_backoff": True,
        "base_retry_delay": 1.0,
        "max_retry_delay": 30.0,
        "enable_human_review": True,
    }


@pytest.fixture
def empty_metrics():
    """Empty guardrail metrics."""
    return {
        "tool_calls_this_turn": 0,
        "total_tool_calls": 0,
        "tokens_last_minute": 0,
        "retry_count": 0,
        "violations": [],
        "last_token_reset": 0,
        "trace_id": "",
    }
