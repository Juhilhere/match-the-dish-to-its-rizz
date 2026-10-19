import numpy as np
from scipy.special import logsumexp

def normalize_scores(S, mode='friend'):
    if mode == 'friend':
        return (S - S.mean(axis=1, keepdims=True)) / (S.std() + 1e-8)
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-8)

def sinkhorn_log(S, temperature=0.095, iterations=100):
    M = S / temperature
    for _ in range(iterations):
        M = M - logsumexp(M, axis=1, keepdims=True)
        M = M - logsumexp(M, axis=0, keepdims=True)
    return M
