from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import os


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    rup = 4
    lup = 5
    ldown = 6
    rdown = 7


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=6):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the_mouse's and the cheese's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Dict(
            {
                "cat": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "cheese": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
                "mouse": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # We have 8 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = {
        #     Actions.right.value: np.array([1, 0]),
        #     Actions.up.value: np.array([0, 1]),
        #     Actions.left.value: np.array([-1, 0]),
        #     Actions.down.value: np.array([0, -1]),
        #     Actions.rup.value: np.array([1, 1]),
        #     Actions.lup.value: np.array([-1, 1]),
        #     Actions.ldown.value: np.array([-1, -1]),
        #     Actions.rdown.value: np.array([1, -1]),
        # }

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),    # Decreasing y moves UP in Pygame
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),     # Increasing y moves DOWN in Pygame
            Actions.rup.value: np.array([1, -1]),
            Actions.lup.value: np.array([-1, -1]),
            Actions.ldown.value: np.array([-1, 1]),
            Actions.rdown.value: np.array([1, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
                "cat": self._cat_location, 
                "cheese": self._cheese_location, 
                "mouse": self._mouse_location,
                }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._cheese_location - self._mouse_location, ord=1
            ),
            "cat_distance": np.linalg.norm(
                self._mouse_location - self._cat_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the_mouse's location uniformly at random
        self._mouse_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the cheese's location randomly until it does not
        # coincide with the_mouse's location
        self._cheese_location = self._mouse_location
        while np.array_equal(self._cheese_location, self._mouse_location):
            self._cheese_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
            
        self._cat_location = self._mouse_location
        while np.linalg.norm(self._cat_location - self._mouse_location, ord=1) < 3:
            self._cat_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated = False
        ##current dist to cheese
        old_dist = self._get_info()["distance"]
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._mouse_location = np.clip(
            self._mouse_location + direction, 0, self.size - 1
        )
        new_dist = self._get_info()["distance"]
        # An episode is done iff the_mouse has reached the cheese
        mouseWin = np.array_equal(self._mouse_location, self._cheese_location)
        catWin = np.array_equal(self._mouse_location, self._cat_location)
        if not mouseWin and not catWin:
            cat_direction = np.sign(self._mouse_location - self._cat_location)
            self._cat_location = np.clip(
                self._cat_location + cat_direction, 0, self.size - 1
            )
            catWin = np.array_equal(self._mouse_location, self._cat_location)
        
        # REWARD Model
        reward = 0.0
        
        if mouseWin:
            reward = 20.0
            terminated = True
        elif catWin:
            reward = -10.0
            terminated = True
        
        reward += 0.2 * (old_dist - new_dist)
            
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / self.size

        # Path to assets folder
        asset_path = os.path.join(os.path.dirname(__file__), "resources")

        # Load images
        self.cat_image = pygame.image.load(
            os.path.join(asset_path, "cat.png")
        )

        self.mouse_image = pygame.image.load(
            os.path.join(asset_path, "mouse.png")
        )

        self.cheese_image = pygame.image.load(
            os.path.join(asset_path, "cheese.png")
        )
        
        # Scale images to one grid cell
        cat_img = pygame.transform.scale(
            self.cat_image,
            (int(pix_square_size), int(pix_square_size))
        )

        mouse_img = pygame.transform.scale(
            self.mouse_image,
            (int(pix_square_size), int(pix_square_size))
        )

        cheese_img = pygame.transform.scale(
            self.cheese_image,
            (int(pix_square_size), int(pix_square_size))
        )

        # Draw cheese
        canvas.blit(
            cheese_img,
            (
                self._cheese_location[0] * pix_square_size,
                self._cheese_location[1] * pix_square_size,
            ),
        )

        # Draw mouse
        canvas.blit(
            mouse_img,
            (
                self._mouse_location[0] * pix_square_size,
                self._mouse_location[1] * pix_square_size,
            ),
        )

        # Draw cat
        canvas.blit(
            cat_img,
            (
                self._cat_location[0] * pix_square_size,
                self._cat_location[1] * pix_square_size,
            ),
        )

        # Draw grid
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )

            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
