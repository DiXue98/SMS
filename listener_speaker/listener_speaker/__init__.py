from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 20)
players = range(2, 10)

for s, p in product(sizes, players):
    register(
        id="ListenerSpeaker-{0}x{0}-{1}-v1".format(s, p),
        entry_point="listener_speaker.listener_speaker_env:ListenerSpeakerEnv",
        kwargs={
            "players": p,
            "field_size": (s, s),
            "max_episode_steps": 50,
        },
    )