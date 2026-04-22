from __future__ import annotations

import numpy as np


def make_unitary(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    spectrum = np.fft.fft(vec)
    magnitudes = np.abs(spectrum)
    magnitudes[magnitudes < eps] = 1.0
    return np.fft.ifft(spectrum / magnitudes).real


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def bind(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Circular convolution binding."""
    return np.fft.ifft(np.fft.fft(left) * np.fft.fft(right)).real


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Approximate circular-correlation unbinding."""
    return np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(key))).real


def cosine(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom < eps:
        return 0.0
    return float(np.dot(left, right) / denom)
