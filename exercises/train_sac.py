# train_sac.py
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from rip_env import RotaryInvertedPendulumEnv

def make_env():
    env = RotaryInvertedPendulumEnv(
        dt=0.01,
        episode_seconds=5.0,
        u_max=3.0,
        # reward shaping (Startwerte)
        w_theta=12.0,
        w_phi=2.5,
        w_dtheta=0.6,
        w_dphi=0.25,
        w_u=0.02,
        upright_bonus=1.0,
        upright_tol=0.12,
    )
    return Monitor(env)

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    # Sehr empfehlenswert: Normalisierung von Observations/Rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(total_timesteps=300_000)

    model.save("sac_rip_policy")
    env.save("vecnormalize_stats.pkl")  # wichtig fürs spätere Abrufen!