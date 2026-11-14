"""Weekly operations report helpers (task 4.8 · R4.5)."""

from __future__ import annotations

from datetime import date, timedelta

from finint.eval.ops_report import simulate_history


def test_observed_days_are_kept_and_others_marked_simulated() -> None:
    observed = {date(2026, 10, 27): 120, date(2026, 10, 28): 240, date(2026, 10, 29): 200}
    history = simulate_history(observed, date(2026, 11, 13), 30, noise=0.1, seed=1)
    assert len(history) == 30
    for day, count in observed.items():
        assert history[day] == (count, False)
    simulated = [v for d, (v, s) in history.items() if s]
    assert len(simulated) == 27
    assert all(0.9 * 200 - 1 <= v <= 1.1 * 240 + 1 for v in simulated)  # full days only, +/- noise


def test_history_is_reproducible_with_a_seed() -> None:
    observed = {date(2026, 10, 28): 240, date(2026, 10, 29): 200}
    end = date(2026, 11, 13)
    first = simulate_history(observed, end, 60, 0.1, 7)
    assert first == simulate_history(observed, end, 60, 0.1, 7)
    assert max(first) == end
    assert min(first) == end - timedelta(days=59)
