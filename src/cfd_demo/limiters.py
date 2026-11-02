"""Selectable scalar TVD limiter functions phi(r) (ADR 0004).

The functions are pure JAX and compatible with ``jit``, ``vmap``, and
``grad``. ADR 0010 defines how the solver applies them.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

LimiterFunction = Callable[[jnp.ndarray], jnp.ndarray]


def minmod(r):
    """phi(r) = max(0, min(1, r))."""
    return jnp.maximum(0.0, jnp.minimum(1.0, r))


def mc(r):
    """Monotonized central: phi(r) = max(0, min(2r, (1 + r)/2, 2))."""
    return jnp.maximum(0.0, jnp.minimum(jnp.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))


def van_leer(r):
    """phi(r) = (r + |r|) / (1 + |r|)."""
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


def superbee(r):
    """phi(r) = max(0, min(2r, 1), min(r, 2))."""
    return jnp.maximum(0.0, jnp.maximum(jnp.minimum(2.0 * r, 1.0), jnp.minimum(r, 2.0)))


def unlimited(r):
    """phi(r) = 1: explicit no-limiting option."""
    r = jnp.asarray(r)
    return jnp.ones_like(r, dtype=jnp.result_type(r, 1.0))


LIMITER_FUNCTIONS: dict[str, LimiterFunction] = {
    "minmod": minmod,
    "mc": mc,
    "van_leer": van_leer,
    "superbee": superbee,
    "unlimited": unlimited,
}


def get_limiter_function(name: str) -> LimiterFunction:
    """Return the limiter function with the given stable name."""
    try:
        return LIMITER_FUNCTIONS[name]
    except KeyError:
        options = ", ".join(sorted(LIMITER_FUNCTIONS))
        raise ValueError(f"Unknown limiter {name!r}; choose one of: {options}.") from None
