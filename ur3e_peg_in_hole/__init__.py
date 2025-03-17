# Import the registration function from Gymnasium
from gymnasium.envs.registration import register

register(
    id="ur3e_peg_in_hole/UR3ePegInHoleEnv-v0",
    entry_point="ur3e_peg_in_hole.envs:UR3ePegInHoleEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)