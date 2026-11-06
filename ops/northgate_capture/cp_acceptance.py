"""Gate CP capture against an already verified TP capture of the same trace.

This short-input comparison does not establish long-input repeatability. The
thresholds are fixed before observing CP results; failures stop all collectors.
"""
import argparse
import time
from pathlib import Path


def compare_tensors(actual, reference):
    import torch

    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('CP reference shape/dtype mismatch')
    a = actual.reshape(-1)
    b = reference.reshape(-1)
    dot = norm_a = norm_b = error = maximum = 0.0
    for start in range(0, a.numel(), 1 << 20):
        left = a[start:start + (1 << 20)].double()
        right = b[start:start + (1 << 20)].double()
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError('nonfinite CP comparison')
        delta = left - right
        dot += (left * right).sum().item()
        norm_a += left.square().sum().item()
        norm_b += right.square().sum().item()
        error += delta.square().sum().item()
        maximum = max(maximum, delta.abs().max().item())
    relative = (error / max(norm_b, 1e-300)) ** 0.5
    cosine = dot / max((norm_a * norm_b) ** 0.5, 1e-300)
    return {'relative_l2': relative, 'cosine': cosine, 'max_abs': maximum,
            'passed': relative <= 0.01 and cosine >= 0.99995}


def compare_features(actual, reference):
    import torch

    for key in ['input_ids', 'loss_mask']:
        if not torch.equal(actual[key], reference[key]):
            raise ValueError(f'CP reference {key} mismatch')
    result = {key: compare_tensors(actual[key], reference[key]) for key in
              ['hidden_states', 'target_last_hidden_states']}
    return {'thresholds': {'max_relative_l2': 0.01, 'min_cosine': 0.99995},
            'tensors': result, 'passed': all(v['passed'] for v in result.values())}


def wait_for_acceptance(run):
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        if (run / 'CP_REJECTED.json').exists():
            raise RuntimeError('CP numerical reference gate rejected; inspect CP_REJECTED.json')
        if (run / 'CP_ACCEPTED.json').exists():
            return
        if (run / 'worker-0/exit.txt').exists():
            raise RuntimeError('designated CP reference worker exited before acceptance')
        time.sleep(5)
    raise TimeoutError('CP reference gate timed out')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', type=Path, required=True)
    wait_for_acceptance(parser.parse_args().wait)
