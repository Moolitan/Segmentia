from validate_prefix_correction import evaluate_gate


def _pair_summary(values, layers):
    rows = []
    for index, (value, improved_layers) in enumerate(zip(values, layers, strict=True)):
        rows.append(
            {
                "case_id": f"case-{index}",
                "variant": "online_prefix_k",
                "aggregate_improvement_vs_direct": value,
                "improved_layers": improved_layers,
            }
        )
    return rows


def _offset_rows(values):
    return [
        {"case_id": f"case-{index // 40}", "online_offline_offset_cosine": value}
        for index, value in enumerate(values)
    ]


def test_prefix_gate_go():
    gate = evaluate_gate(
        _pair_summary([0.20, 0.25, 0.30], [35, 38, 40]),
        _offset_rows([0.995] * 120),
        {"external_apply_events": 0},
    )

    assert gate["status"] == "go"
    assert gate["positive_pairs"] == 3


def test_prefix_gate_rejects_weak_worst_case():
    gate = evaluate_gate(
        _pair_summary([0.04, 0.25, 0.30], [35, 38, 40]),
        _offset_rows([0.995] * 120),
        {"external_apply_events": 0},
    )

    assert gate["status"] == "no_go"
    assert gate["worst_improvement"] == 0.04


def test_prefix_gate_rejects_offset_mismatch():
    gate = evaluate_gate(
        _pair_summary([0.20, 0.25, 0.30], [35, 38, 40]),
        _offset_rows([0.98] * 120),
        {"external_apply_events": 0},
    )

    assert gate["status"] == "no_go"
    assert gate["median_offset_cosine"] == 0.98


def test_prefix_gate_requires_short_fallback():
    gate = evaluate_gate(
        _pair_summary([0.20, 0.25, 0.30], [35, 38, 40]),
        _offset_rows([0.995] * 120),
        None,
    )

    assert gate["status"] == "no_go"
    assert gate["short_full_fallback_passed"] is False


def test_prefix_gate_requires_three_pairs():
    gate = evaluate_gate(
        _pair_summary([0.20, 0.25], [35, 38]),
        _offset_rows([0.995] * 80),
        {"external_apply_events": 0},
    )

    assert gate["status"] == "insufficient_pairs"
