from gym.envs.registration import register

register(
    id='eliftech-v0',
    entry_point='gym_eliftech_racing.envs: RacingSimpleEnv'
)
