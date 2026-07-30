from validate_cap4_followup import build_comparison


def test_build_comparison_separates_cap_and_materialized_baselines():
    baseline = {
        "points": [
            {
                "shape": "long-8k",
                "mode": "materialized",
                "followers": 4,
                "wall_s": 1.6,
                "throughput_req_s": 2.5,
            },
            {
                "shape": "long-8k",
                "mode": "shared",
                "followers": 4,
                "wall_s": 2.4,
                "throughput_req_s": 1.666667,
            },
        ]
    }
    cap4 = {"wall_s": 1.8, "throughput_req_s": 2.222222}

    row = build_comparison("long-8k", cap4, baseline)

    assert row["cap4_vs_cap2_wall_speedup"] == 2.4 / 1.8
    assert row["cap4_vs_materialized_wall_ratio"] == 1.8 / 1.6
    assert row["cap4_wall_s"] == 1.8
