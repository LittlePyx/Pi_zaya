from __future__ import annotations

import threading

from kb.converter.vision_circuit_breaker import (
    VisionCircuitBreaker,
    VisionCircuitPolicy,
    build_vision_circuit_key,
    load_vision_circuit_policy,
    vision_circuit_failure_kind,
)


def _policy(
    *,
    threshold: int = 2,
    cooldown_s: float = 10.0,
    ttl_s: float = 30.0,
    max_entries: int = 8,
) -> VisionCircuitPolicy:
    return VisionCircuitPolicy(
        enabled=True,
        failure_threshold=threshold,
        cooldown_s=cooldown_s,
        state_ttl_s=ttl_s,
        max_entries=max_entries,
    )


def test_circuit_opens_half_opens_and_recovers_after_success():
    now = [100.0]
    breaker = VisionCircuitBreaker(clock=lambda: now[0])
    key = build_vision_circuit_key(
        base_url="https://dashscope.example/v1",
        model="vision-model",
        speed_mode="ultra_fast",
    )
    policy = _policy(threshold=2, cooldown_s=10.0)

    assert breaker.before_request(key, policy).allow_request is True
    assert breaker.record_failure(key, failure_kind="timeout", policy=policy) is False
    assert breaker.before_request(key, policy).allow_request is True
    assert breaker.record_failure(key, failure_kind="rate_limited", policy=policy) is True

    opened = breaker.before_request(key, policy)
    assert opened.allow_request is False
    assert opened.reason == "open"
    assert opened.retry_after_s == 10.0

    now[0] += 10.1
    probe = breaker.before_request(key, policy)
    assert probe.allow_request is True
    assert probe.reason == "half_open_probe"
    assert breaker.before_request(key, policy).reason == "probe_inflight"

    breaker.record_success(key)
    recovered = breaker.before_request(key, policy)
    assert recovered.allow_request is True
    assert recovered.reason == "closed"


def test_neutral_business_error_breaks_failure_sequence_and_releases_probe():
    now = [10.0]
    breaker = VisionCircuitBreaker(clock=lambda: now[0])
    key = ("https://provider.example/v1", "vision-model", "normal")
    policy = _policy(threshold=2, cooldown_s=5.0)

    breaker.record_failure(key, failure_kind="timeout", policy=policy)
    breaker.record_neutral(key)
    assert breaker.record_failure(key, failure_kind="timeout", policy=policy) is False
    assert breaker.before_request(key, policy).allow_request is True


def test_half_open_allows_only_one_concurrent_probe():
    now = [20.0]
    breaker = VisionCircuitBreaker(clock=lambda: now[0])
    key = ("https://provider.example/v1", "vision-model", "ultra_fast")
    policy = _policy(threshold=1, cooldown_s=5.0)
    breaker.record_failure(key, failure_kind="timeout", policy=policy)
    now[0] += 5.1

    decisions = []
    decisions_lock = threading.Lock()
    start = threading.Barrier(9)

    def decide():
        start.wait(timeout=2.0)
        result = breaker.before_request(key, policy)
        with decisions_lock:
            decisions.append(result)

    threads = [threading.Thread(target=decide) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait(timeout=2.0)
    for thread in threads:
        thread.join(timeout=2.0)

    assert len(decisions) == 8
    assert sum(1 for item in decisions if item.allow_request) == 1
    assert sum(1 for item in decisions if item.reason == "probe_inflight") == 7


def test_circuit_state_is_bounded_and_short_lived():
    now = [0.0]
    breaker = VisionCircuitBreaker(clock=lambda: now[0])
    policy = _policy(threshold=3, cooldown_s=10.0, ttl_s=5.0, max_entries=4)

    for index in range(10):
        key = ("https://provider.example/v1", f"vision-{index}", "normal")
        breaker.record_failure(key, failure_kind="timeout", policy=policy)
        now[0] += 0.1
    assert breaker.state_count() == 4

    now[0] += 6.0
    breaker.before_request(("https://other.example/v1", "vision", "normal"), policy)
    assert breaker.state_count() == 0


def test_normal_and_ultra_fast_policies_are_independently_configurable(monkeypatch):
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_BREAKER", "1")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_NORMAL_THRESHOLD", "5")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_NORMAL_COOLDOWN_S", "12")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_ULTRA_FAST_THRESHOLD", "1")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_ULTRA_FAST_COOLDOWN_S", "40")

    normal = load_vision_circuit_policy("normal")
    ultra = load_vision_circuit_policy("ultra_fast")

    assert (normal.failure_threshold, normal.cooldown_s) == (5, 12.0)
    assert (ultra.failure_threshold, ultra.cooldown_s) == (1, 40.0)

    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_BREAKER", "0")
    assert load_vision_circuit_policy("normal").enabled is False
    assert load_vision_circuit_policy("ultra_fast").enabled is False

    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_BREAKER", "1")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_NORMAL_THRESHOLD", "0")
    assert load_vision_circuit_policy("normal").enabled is False


def test_disabling_circuit_immediately_clears_existing_open_state():
    now = [30.0]
    breaker = VisionCircuitBreaker(clock=lambda: now[0])
    key = ("https://provider.example/v1", "vision-model", "normal")
    enabled = _policy(threshold=1, cooldown_s=30.0)
    disabled = VisionCircuitPolicy(
        enabled=False,
        failure_threshold=1,
        cooldown_s=1.0,
        state_ttl_s=5.0,
        max_entries=8,
    )

    breaker.record_failure(key, failure_kind="timeout", policy=enabled)
    assert breaker.before_request(key, enabled).allow_request is False
    assert breaker.before_request(key, disabled).allow_request is True
    assert breaker.before_request(key, enabled).reason == "closed"


def test_failure_classifier_only_counts_timeout_and_rate_limit():
    class RateLimitError(RuntimeError):
        status_code = 429

    class BusinessError(RuntimeError):
        status_code = 422

    assert vision_circuit_failure_kind(TimeoutError("hard timeout")) == "timeout"
    assert vision_circuit_failure_kind(RateLimitError("quota")) == "rate_limited"
    assert vision_circuit_failure_kind(RuntimeError("Error code: 429, too many requests")) == "rate_limited"
    assert vision_circuit_failure_kind(BusinessError("invalid document")) == ""


def test_circuit_key_drops_credentials_query_and_separates_speed_modes():
    normal = build_vision_circuit_key(
        base_url="https://user:secret@Provider.Example/v1/?token=hidden",
        model="Vision-Model",
        speed_mode="normal",
    )
    ultra = build_vision_circuit_key(
        base_url="https://provider.example/v1",
        model="vision-model",
        speed_mode="ultra_fast",
    )

    assert normal == ("https://provider.example/v1", "vision-model", "normal")
    assert "secret" not in normal[0]
    assert normal[:2] == ultra[:2]
    assert normal[2] != ultra[2]
