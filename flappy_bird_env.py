import pygame
import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self, render_mode=False):
        pygame.init()
        self.WIDTH, self.HEIGHT = 400, 600
        self.gravity = 0.5
        self.flap_strength = -8
        self.pipe_gap = 150
        self.pipe_width = 60
        self.render_mode = render_mode

        if render_mode:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.bird_y = self.HEIGHT // 2
        self.bird_vel = 0
        self.pipe_x = self.WIDTH
        self.pipe_height = random.randint(100, 400)
        self.score = 0
        return self._get_state()

    def step(self, action):
        # Action: 0 = do nothing, 1 = flap
        if action == 1:
            self.bird_vel = self.flap_strength
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        self.pipe_x -= 4
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.WIDTH
            self.pipe_height = random.randint(100, 400)
            self.score += 1

        done = False

        # Reward shaping: stay near center of pipe gap
        pipe_center = self.pipe_height + self.pipe_gap / 2
        distance = abs(self.bird_y - pipe_center)
        reward = 1.0 - 0.01 * distance  # Max ~1, drops as bird moves away

        if self.check_collision():
            reward = -100.0
            done = True

        state = self._get_state()
        return state, reward, done

    def _get_state(self):
        return np.array([
            self.bird_y / self.HEIGHT,
            self.bird_vel / 10,
            (self.pipe_x - 50) / self.WIDTH,
            self.pipe_height / self.HEIGHT
        ], dtype=np.float32)

    def check_collision(self):
        # Check for ground or ceiling hit
        if self.bird_y < 0 or self.bird_y > self.HEIGHT:
            return True
        # Check for pipe collision (bird is at x = 50, radius = 15)
        if 50 + 15 > self.pipe_x and 50 - 15 < self.pipe_x + self.pipe_width:
            if self.bird_y - 15 < self.pipe_height or self.bird_y + 15 > self.pipe_height + self.pipe_gap:
                return True
        return False

    def render(self):
        if not self.render_mode:
            return
        self.screen.fill((255, 255, 255))  # White background
        pygame.draw.circle(self.screen, (255, 0, 0), (50, int(self.bird_y)), 15)  # Bird
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, 0, self.pipe_width, self.pipe_height))  # Top pipe
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, self.pipe_height + self.pipe_gap, self.pipe_width, self.HEIGHT))  # Bottom pipe
        pygame.display.update()
        self.clock.tick(30)
