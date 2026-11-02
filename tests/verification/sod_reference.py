"""Independent exact solution of the one-dimensional Sod Riemann problem.

Test-side reference implemented with NumPy only (ADR 0006): it must not reuse
solver kernels. Follows Toro, *Riemann Solvers and Numerical Methods for Fluid
Dynamics*, 3rd ed., chapter 4 (the ``STAR_PU`` and ``SAMPLE`` routines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEWTON_TOLERANCE = 1e-14
NEWTON_ITERATIONS = 100


@dataclass(frozen=True)
class RiemannState:
    """Primitive state ``(rho, u, p)`` of a one-dimensional Riemann problem."""

    rho: float
    u: float
    p: float

    def sound_speed(self, gamma: float) -> float:
        return float(np.sqrt(gamma * self.p / self.rho))


@dataclass(frozen=True)
class RiemannSolution:
    """Exact solution of a Riemann problem centred at ``position``.

    Attributes:
        star_pressure, star_velocity: Pressure and velocity of the star region.
        star_density_left, star_density_right: Densities on either side of the
            contact.
        head_speed, tail_speed: Speeds of the left rarefaction head and tail.
        contact_speed: Speed of the contact discontinuity.
        shock_speed: Speed of the right shock.
    """

    left: RiemannState
    right: RiemannState
    gamma: float
    position: float
    star_pressure: float
    star_velocity: float
    star_density_left: float
    star_density_right: float
    head_speed: float
    tail_speed: float
    contact_speed: float
    shock_speed: float

    def wave_positions(self, time: float) -> dict[str, float]:
        """Positions of the rarefaction head and tail, contact, and shock."""
        return {
            "head": self.position + self.head_speed * time,
            "tail": self.position + self.tail_speed * time,
            "contact": self.position + self.contact_speed * time,
            "shock": self.position + self.shock_speed * time,
        }

    def sample(self, x, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Primitive variables ``(rho, u, p)`` at coordinates ``x`` and ``time``."""
        return _sample(self, np.asarray(x, dtype=np.float64), float(time))


def _pressure_function(p: float, state: RiemannState, gamma: float) -> tuple[float, float]:
    """Toro's ``PREFUN``: the pressure function and its derivative."""
    c = state.sound_speed(gamma)
    if p > state.p:  # shock
        a = 2.0 / ((gamma + 1.0) * state.rho)
        b = (gamma - 1.0) / (gamma + 1.0) * state.p
        root = np.sqrt(a / (b + p))
        return (p - state.p) * root, root * (1.0 - 0.5 * (p - state.p) / (b + p))
    ratio = p / state.p  # rarefaction
    exponent = (gamma - 1.0) / (2.0 * gamma)
    return (
        2.0 * c / (gamma - 1.0) * (ratio**exponent - 1.0),
        ratio ** (-(gamma + 1.0) / (2.0 * gamma)) / (state.rho * c),
    )


def solve_riemann(
    left: RiemannState, right: RiemannState, gamma: float = 1.4, position: float = 0.5
) -> RiemannSolution:
    """Solve the Riemann problem exactly with Newton iteration on the star
    pressure. Raises ``ValueError`` if a vacuum would form."""
    c_left = left.sound_speed(gamma)
    c_right = right.sound_speed(gamma)
    if 2.0 * (c_left + c_right) / (gamma - 1.0) <= right.u - left.u:
        raise ValueError("The initial states generate a vacuum.")

    exponent = (gamma - 1.0) / (2.0 * gamma)
    guess = (
        (c_left + c_right - 0.5 * (gamma - 1.0) * (right.u - left.u))
        / (c_left / left.p**exponent + c_right / right.p**exponent)
    ) ** (1.0 / exponent)
    pressure = max(guess, 1e-12)
    for _ in range(NEWTON_ITERATIONS):
        f_left, df_left = _pressure_function(pressure, left, gamma)
        f_right, df_right = _pressure_function(pressure, right, gamma)
        change = (f_left + f_right + right.u - left.u) / (df_left + df_right)
        pressure -= change
        pressure = max(pressure, 1e-12)
        if abs(change) <= NEWTON_TOLERANCE * max(pressure, 1.0):
            break
    else:
        raise RuntimeError("The star-pressure iteration did not converge.")

    f_left, _ = _pressure_function(pressure, left, gamma)
    f_right, _ = _pressure_function(pressure, right, gamma)
    velocity = 0.5 * (left.u + right.u + f_right - f_left)

    ratio_left = pressure / left.p
    if pressure > left.p:  # left shock
        beta = (gamma - 1.0) / (gamma + 1.0)
        density_left = left.rho * (ratio_left + beta) / (beta * ratio_left + 1.0)
        head_speed = tail_speed = left.u - c_left * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio_left + (gamma - 1.0) / (2.0 * gamma)
        )
    else:  # left rarefaction
        density_left = left.rho * ratio_left ** (1.0 / gamma)
        head_speed = left.u - c_left
        tail_speed = velocity - c_left * ratio_left ** ((gamma - 1.0) / (2.0 * gamma))

    ratio_right = pressure / right.p
    if pressure > right.p:  # right shock
        beta = (gamma - 1.0) / (gamma + 1.0)
        density_right = right.rho * (ratio_right + beta) / (beta * ratio_right + 1.0)
        shock_speed = right.u + c_right * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio_right + (gamma - 1.0) / (2.0 * gamma)
        )
    else:  # right rarefaction
        density_right = right.rho * ratio_right ** (1.0 / gamma)
        shock_speed = right.u + c_right * ratio_right ** ((gamma - 1.0) / (2.0 * gamma))

    return RiemannSolution(
        left=left,
        right=right,
        gamma=gamma,
        position=position,
        star_pressure=pressure,
        star_velocity=velocity,
        star_density_left=density_left,
        star_density_right=density_right,
        head_speed=head_speed,
        tail_speed=tail_speed,
        contact_speed=velocity,
        shock_speed=shock_speed,
    )


def _sample(solution: RiemannSolution, x: np.ndarray, time: float):
    """Sample the self-similar solution; ``time`` must be positive."""
    if time <= 0.0:
        raise ValueError("The sampling time must be positive.")
    gamma = solution.gamma
    left, right = solution.left, solution.right
    c_left = left.sound_speed(gamma)
    c_right = right.sound_speed(gamma)
    speed = (x - solution.position) / time

    rho = np.empty_like(speed)
    u = np.empty_like(speed)
    p = np.empty_like(speed)

    # Left of the contact.
    left_of_contact = speed <= solution.star_velocity
    if solution.star_pressure > left.p:  # left shock
        unshocked = left_of_contact & (speed < solution.head_speed)
        shocked = left_of_contact & ~unshocked
        rho[unshocked], u[unshocked], p[unshocked] = left.rho, left.u, left.p
        rho[shocked] = solution.star_density_left
        u[shocked], p[shocked] = solution.star_velocity, solution.star_pressure
    else:  # left rarefaction
        undisturbed = left_of_contact & (speed < solution.head_speed)
        fan = left_of_contact & (speed >= solution.head_speed) & (speed < solution.tail_speed)
        star = left_of_contact & (speed >= solution.tail_speed)
        rho[undisturbed], u[undisturbed], p[undisturbed] = left.rho, left.u, left.p
        factor = 2.0 / (gamma + 1.0) + (gamma - 1.0) / ((gamma + 1.0) * c_left) * (left.u - speed[fan])
        rho[fan] = left.rho * factor ** (2.0 / (gamma - 1.0))
        u[fan] = 2.0 / (gamma + 1.0) * (c_left + 0.5 * (gamma - 1.0) * left.u + speed[fan])
        p[fan] = left.p * factor ** (2.0 * gamma / (gamma - 1.0))
        rho[star] = solution.star_density_left
        u[star], p[star] = solution.star_velocity, solution.star_pressure

    # Right of the contact.
    right_of_contact = ~left_of_contact
    if solution.star_pressure > right.p:  # right shock
        unshocked = right_of_contact & (speed > solution.shock_speed)
        shocked = right_of_contact & ~unshocked
        rho[unshocked], u[unshocked], p[unshocked] = right.rho, right.u, right.p
        rho[shocked] = solution.star_density_right
        u[shocked], p[shocked] = solution.star_velocity, solution.star_pressure
    else:  # right rarefaction
        undisturbed = right_of_contact & (speed > solution.shock_speed)
        fan = right_of_contact & (speed <= solution.shock_speed) & (speed > solution.tail_speed)
        star = right_of_contact & (speed <= solution.tail_speed)
        rho[undisturbed], u[undisturbed], p[undisturbed] = right.rho, right.u, right.p
        factor = 2.0 / (gamma + 1.0) - (gamma - 1.0) / ((gamma + 1.0) * c_right) * (right.u - speed[fan])
        rho[fan] = right.rho * factor ** (2.0 / (gamma - 1.0))
        u[fan] = 2.0 / (gamma + 1.0) * (-c_right + 0.5 * (gamma - 1.0) * right.u + speed[fan])
        p[fan] = right.p * factor ** (2.0 * gamma / (gamma - 1.0))
        rho[star] = solution.star_density_right
        u[star], p[star] = solution.star_velocity, solution.star_pressure

    return rho, u, p


def sod_solution(position: float = 0.5, gamma: float = 1.4) -> RiemannSolution:
    """The exact solution of the standard Sod problem of ADR 0005."""
    return solve_riemann(
        RiemannState(rho=1.0, u=0.0, p=1.0),
        RiemannState(rho=0.125, u=0.0, p=0.1),
        gamma=gamma,
        position=position,
    )
