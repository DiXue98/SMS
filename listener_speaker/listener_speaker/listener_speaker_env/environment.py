import logging
from collections import namedtuple
from enum import Enum
from gym import Env
import gym
from gym.utils import seeding
import numpy as np

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

class Player:
    def __init__(self):
        self.position = None
        self.index = None

    def setup(self, position, index, field_size):
        self.position = position
        self.index = index

class Speaker:
    def __init__(self):
        self.position = None
        self.index = None

    def setup(self, position, index):
        self.position = position
        self.index = index

class ListenerSpeakerEnv(Env):
    action_set = [Action.NONE, Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["actions", "index", "game_over", "current_step"],
    )

    def __init__(
        self,
        players,
        field_size,
        max_episode_steps,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.n_agents = players
        self.n_speakers = self.n_agents
        self.players = [Player() for _ in range(self.n_agents)]
        self.speakers = [Speaker() for _ in range(self.n_speakers)]

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self._field_size = field_size
        self._game_over = None
        self._valid_actions = None
        self.msg_from_speakers = None
        self._max_episode_steps = max_episode_steps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - the one-hot vector of the id of the assigned speakers to the listeners
        """
        min_obs, max_obs = [0] * self.n_agents, [1] * self.n_agents
    
        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @property
    def field_size(self):
        return self._field_size

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }
        self.avail_actions = [[1 if action in self._valid_actions[player] else 0 for action in Action] for player in self.players]

    def spawn_speakers(self):
        speaker_count = 0
        attempts = 0

        speaker_index = list(range(self.n_agents))

        while speaker_count < self.n_speakers and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it is occupied
            if not self._is_empty_location(row, col):
                continue

            self.speakers[speaker_count].setup((row, col), speaker_index[speaker_count])
            speaker_count += 1

    def _is_empty_location(self, row, col):
        # check if occupied by speakers
        for s in self.speakers:
            if s.position and row == s.position[0] and col == s.position[1]:
                return False

        # check if occupied by players
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self):
        self.player_index = list(range(self.n_agents))

        for id, player in enumerate(self.players):
            attempts = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows - 1)
                col = self.np_random.randint(0, self.cols - 1)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.player_index[id],
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return player.position[0] > 0
        elif action == Action.SOUTH:
            return player.position[0] < self.rows - 1
        elif action == Action.WEST:
            return player.position[1] > 0
        elif action == Action.EAST:
            return player.position[1] < self.cols - 1

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            index=player.index,
            game_over=self.game_over,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            obs[observation.index] = 1
            return obs

        def distance(player, speaker):
            dist = (player.position[0] - speaker.position[0]) ** 2 + (player.position[1] - speaker.position[1]) ** 2
            return np.sqrt(dist)

        def get_player_reward(id, observation):
            return -0.01 * distance(self.players[id], self.speakers[self.players[id].index])

        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(id, obs) for id, obs in enumerate(observations)]
        ndone = [obs.game_over for obs in observations]
        ninfo = {}

        return nobs, nreward, ndone, ninfo

    def make_speaker_obs(self):
        speaker_obs = []
        for i in range(self.n_speakers):
            obs = []
            for player in self.players:
                if player.index == i:
                    obs += [self.speakers[i].position[0]-player.position[0],
                            self.speakers[i].position[1]-player.position[1]]
                    break
            speaker_obs.append(np.array(obs))
        return speaker_obs
    
    def _get_msg(self):
        msg = []
        for i in range(self.n_speakers):
            obs = []
            for player in self.players:
                obs += [self.speakers[i].position[0]-player.position[0],
                        self.speakers[i].position[1]-player.position[1]]

            msg.append(np.array(obs))
        return msg

    def reset(self):
        self.spawn_players()
        self.spawn_speakers()

        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        self.msg_from_speakers = self._get_msg()
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        return nobs

    def step(self, actions):
        self.current_step += 1

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # move players
        for player, action in zip(self.players, actions):
            if action == Action.NORTH:
                player.position = (player.position[0] - 1, player.position[1])
            elif action == Action.SOUTH:
                player.position = (player.position[0] + 1, player.position[1])
            elif action == Action.WEST:
                player.position = (player.position[0], player.position[1] - 1)
            elif action == Action.EAST:
                player.position = (player.position[0], player.position[1] + 1)

        self._game_over = (self._max_episode_steps <= self.current_step)
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        self.msg_from_speakers = self._get_msg()
        return self._make_gym_obs(observations)