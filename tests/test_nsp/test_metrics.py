import pytest

from specforge.nsp.metrics import MetricTotals, parse_bins
from specforge.nsp.serving_eval import serving_terms


def test_bins_pool_counts_and_keep_empty_bins_null():
    totals = MetricTotals(["accuracy", "al"], parse_bins("0,4k,8k,16k,inf"))
    totals.add({"accuracy": (1, 1), "al": (8, 2)}, ctx_len=4095)
    totals.add({"accuracy": (0, 9), "al": (12, 6)}, ctx_len=4096)
    result = totals.result()
    assert result["overall"]["accuracy"]["value"] == 0.1
    assert result["overall"]["al"]["value"] == 2.5
    assert result["bins"][0]["metrics"]["accuracy"]["value"] == 1
    assert result["bins"][1]["metrics"]["accuracy"]["value"] == 0
    assert result["bins"][2]["metrics"]["accuracy"]["value"] is None


@pytest.mark.parametrize(
    "edges", ["0,4k,4k,inf", "1,4k,inf", "0,8k", "0,nan,inf", "0,1.5,inf"]
)
def test_invalid_bins_fail(edges):
    with pytest.raises(ValueError):
        parse_bins(edges)


def test_request_al_is_committed_tokens_not_one_plus_correct_drafts():
    rows = serving_terms(
        {"spec_verify_ct": 67, "spec_num_correct_drafts": 223}, 287, False
    )
    assert rows == [(None, 287, 67)]
    with pytest.raises(ValueError, match="spec_verify_cycles"):
        serving_terms({"spec_verify_ct": 67}, 287, True)
    with pytest.raises(ValueError, match="spec_verify_ct"):
        serving_terms({}, 287, False)


def test_cycle_telemetry_is_exact_and_duplicate_cycles_rejected():
    cycles = [
        {"cycle_id": 1, "ctx_len": 4095, "committed_output_tokens": 3},
        {"cycle_id": 2, "ctx_len": 4098, "committed_output_tokens": 1},
    ]
    assert serving_terms({"spec_verify_cycles": cycles}, 4, True) == [
        (4095, 3, 1),
        (4098, 1, 1),
    ]
    with pytest.raises(ValueError, match="disagrees"):
        serving_terms({"spec_verify_cycles": cycles}, 5, True)
    with pytest.raises(ValueError, match="duplicate"):
        serving_terms({"spec_verify_cycles": [cycles[0], cycles[0]]}, 6, True)
