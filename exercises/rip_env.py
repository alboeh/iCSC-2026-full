# rip_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RotaryInvertedPendulumEnv(gym.Env):
    """
    Observation: [theta, dtheta, phi, dphi]
    Action: u = ddphi (rotary acceleration), continuous, clipped to [-u_max, u_max]
    Upright target: theta = 0, phi = 0
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.01,
        episode_seconds=5.0,
        u_max=3.0,
        # pendulum params (deine)
        g=9.81,
        l=0.235,
        r=0.11,
        b=2e-4,
        total_mass_grams=20.8,
        # reward weights
        w_theta=10.0,     # STRONG
        w_phi=2.0,        # MEDIUM
        w_dtheta=0.5,     # speed penalty
        w_dphi=0.2,       # speed penalty
        w_u=0.01,         # action smoothness
        upright_bonus=1.0,
        upright_tol=0.12, # rad ~ 7°
        # termination
        theta_fail=np.pi/2,   # fail if pendulum falls too far
        phi_wrap=True,
        seed=None,
    ):
        super().__init__()

        self.dt = float(dt)
        self.episode_seconds = float(episode_seconds)
        self.max_steps = int(np.round(self.episode_seconds / self.dt))
        self.u_max = float(u_max)

        self.g = float(g)
        self.l = float(l)
        self.r = float(r)
        self.b = float(b)

        # Masse nur für vertikalen Teil wie bei dir
        total_mass = (total_mass_grams / 1000.0)
        self.m = total_mass * self.l / (self.r + self.l)
        self.J = (1.0 / 3.0) * self.m * self.l**2

        # reward
        self.w_theta = float(w_theta)
        self.w_phi = float(w_phi)
        self.w_dtheta = float(w_dtheta)
        self.w_dphi = float(w_dphi)
        self.w_u = float(w_u)
        self.upright_bonus = float(upright_bonus)
        self.upright_tol = float(upright_tol)

        # termination
        self.theta_fail = float(theta_fail)
        self.phi_wrap = bool(phi_wrap)

        # action/observation spaces
        self.action_space = spaces.Box(
            low=np.array([-self.u_max], dtype=np.float32),
            high=np.array([ self.u_max], dtype=np.float32),
            dtype=np.float32
        )

        # Observation bounds: hier bewusst "weit" (unbounded-ish), RL algos kommen klar.
        high = np.array([np.pi, 50.0, np.pi, 200.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self.state = np.zeros(4, dtype=np.float64)

    def _wrap_angle(self, a):
        # [-pi, pi]
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _dynamics_step(self, state, u):
        theta, dtheta, phi, dphi = state
        dt = self.dt

        # your ODE
        ddphi = u
        ddtheta = (
            (-self.m * self.r * self.l / 2.0 * ddphi)
            + (self.m * self.l / 2.0 * (self.r + self.l/2.0*np.sin(theta)) * np.cos(theta) * dphi**2)
            + (self.m * self.g * self.l / 2.0 * np.sin(theta))
            - (self.b * dtheta)
        ) / self.J

        # semi-implicit Euler
        dtheta_new = dtheta + dt * ddtheta
        theta_new  = theta  + dt * dtheta_new
        dphi_new   = dphi   + dt * ddphi
        phi_new    = phi    + dt * dphi_new

        if self.phi_wrap:
            phi_new = self._wrap_angle(phi_new)

        # (theta kann man auch wrappen, aber fürs Fail-Kriterium ist "echte" Abweichung praktisch)
        theta_new = self._wrap_angle(theta_new)

        return np.array([theta_new, dtheta_new, phi_new, dphi_new], dtype=np.float64)

    def _reward(self, s, u):
        theta, dtheta, phi, dphi = s

        # Quadratische Kosten (klassisch, stabil)
        cost = (
            self.w_theta  * (theta**2)
            + self.w_phi    * (phi**2)
            + self.w_dtheta * (dtheta**2)
            + self.w_dphi   * (dphi**2)
            + self.w_u      * (u**2)
        )

        # Bonus wenn wirklich nahe upright & phi nahe 0
        bonus = 0.0
        if (abs(theta) < self.upright_tol) and (abs(phi) < 2.0*self.upright_tol):
            bonus = self.upright_bonus

        # Reward = -cost + bonus
        return float(-cost + bonus)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0

        # Initialzustand: kleine Störung um upright (oder größer, wenn du Swing-Up willst)
        # Für Balance-only: klein halten.
        theta0  = self._rng.normal(0.0, 0.08)
        dtheta0 = self._rng.normal(0.0, 0.2)
        phi0    = self._rng.normal(0.0, 0.10)
        dphi0   = self._rng.normal(0.0, 0.5)

        self.state = np.array([theta0, dtheta0, phi0, dphi0], dtype=np.float64)
        obs = self.state.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        self._step_count += 1

        u = float(np.clip(action[0], -self.u_max, self.u_max))
        self.state = self._dynamics_step(self.state, u)
        obs = self.state.astype(np.float32)

        reward = self._reward(self.state, u)

        theta = float(self.state[0])
        terminated = abs(theta) > self.theta_fail  # pendulum "gefallen"
        truncated = self._step_count >= self.max_steps

        info = {"u": u}
        return obs, reward, terminated, truncated, info