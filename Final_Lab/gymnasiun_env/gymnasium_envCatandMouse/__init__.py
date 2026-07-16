from gymnasium.envs.registration import register

register(
    id="gymnasium_envCatandMouse/GridWorld-v0",
    entry_point="gymnasium_envCatandMouse.envs:GridWorldEnv",
    max_episode_steps=20,
)
