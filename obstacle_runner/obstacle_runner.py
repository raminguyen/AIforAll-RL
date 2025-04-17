import pygame
import random
import numpy as np
import pickle
import os
import sys
import datetime

# ==== CONFIG ====
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 100
GROUND_Y = SCREEN_HEIGHT - 70
GRAVITY = 1
JUMP_VELOCITY = -15
OBSTACLE_SPEED = 7
OBSTACLE_FREQ = 90

ACTIONS = [0, 1, 2]
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.01
DISTANCE_BINS = 10
MAX_DISTANCE = 600
Q_table = {}
EPISODE_OFFSET = 0

# ==== COLORS ====
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_TOP = (30, 30, 60)
SKY_BOTTOM = (70, 70, 150)
GREEN = (0, 180, 60)
RED = (255, 60, 60)
BLUE = (50, 200, 255)

# ==== LOAD Q-TABLE ====
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
    print("✅ Q-table loaded. States:", len(Q_table))
else:
    print("⚠️ No previous Q-table found. Starting from scratch.")

# ==== STATE HELPERS ====
def discretize(value, bins, max_val):
    ratio = min(max_val, value) / max_val
    return int(ratio * (bins - 1))

def get_Q(state):
    if state not in Q_table:
        Q_table[state] = [0.0 for _ in ACTIONS]
    return Q_table[state]

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return int(np.argmax(get_Q(state)))

def get_state(agent, obstacle):
    if obstacle:
        distance = obstacle.rect.left - agent.rect.right
        dist_bin = discretize(distance, DISTANCE_BINS, MAX_DISTANCE)
        obs_type = 0 if obstacle.type == "low" else 1
    else:
        dist_bin = DISTANCE_BINS - 1
        obs_type = 2

    if agent.duck_timer > 0:
        agent_state = 2
    elif not agent.on_ground():
        agent_state = 1
    else:
        agent_state = 0

    return (dist_bin, obs_type, agent_state)

# ==== AGENT & OBSTACLE ====
class Agent:
    def __init__(self):
        self.x = 50
        self.width = 40
        self.height = 60
        self.y = GROUND_Y - self.height
        self.color = BLUE
        self.vel_y = 0
        self.duck_timer = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def on_ground(self):
        return self.y >= GROUND_Y - self.height and self.vel_y == 0

    def jump(self):
        if self.on_ground() and self.duck_timer == 0:
            self.vel_y = JUMP_VELOCITY

    def duck(self):
        if self.on_ground():
            self.duck_timer = 15

    def update(self):
        if not self.on_ground():
            self.vel_y += GRAVITY
            self.y += self.vel_y
            if self.y >= GROUND_Y - self.height:
                self.y = GROUND_Y - self.height
                self.vel_y = 0

        if self.duck_timer > 0:
            self.duck_timer -= 1

        draw_height = self.height
        draw_y = self.y

        if self.duck_timer > 0:
            draw_height = int(self.height * 0.5)
            draw_y = self.y + self.height - draw_height

        self.rect = pygame.Rect(self.x, draw_y, self.width, draw_height)

class Obstacle:
    def __init__(self):
        self.width = 30
        self.type = random.choice(["low", "high"])
        if self.type == "low":
            self.height = 40
            self.y = GROUND_Y - self.height
        else:
            self.height = 30
            self.y = GROUND_Y - 150
        self.x = SCREEN_WIDTH
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        self.x -= OBSTACLE_SPEED
        self.rect.x = self.x

# ==== GAME CLASS ====
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("✨ Smooth Agent Runner")
        self.font = pygame.font.SysFont("consolas", 28)
        self.clock = pygame.time.Clock()
        self.reset()

    def draw_gradient_background(self):
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            r = int(SKY_TOP[0] * (1 - ratio) + SKY_BOTTOM[0] * ratio)
            g = int(SKY_TOP[1] * (1 - ratio) + SKY_BOTTOM[1] * ratio)
            b = int(SKY_TOP[2] * (1 - ratio) + SKY_BOTTOM[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def reset(self):
        self.agent = Agent()
        self.obstacles = []
        self.frames_since_obstacle = 0
        self.score = 0
        self.game_over = False

    def spawn_obstacle(self):
        self.obstacles.append(Obstacle())

    def update(self, action):
        if action == 1:
            self.agent.jump()
        elif action == 2:
            self.agent.duck()

        self.agent.update()
        for o in self.obstacles:
            o.update()
        self.obstacles = [o for o in self.obstacles if o.rect.right > 0]

        self.frames_since_obstacle += 1
        if self.frames_since_obstacle >= OBSTACLE_FREQ:
            self.spawn_obstacle()
            self.frames_since_obstacle = 0

        collided = any(self.agent.rect.colliderect(o.rect) for o in self.obstacles)
        self.score += 1
        if collided:
            self.game_over = True
            return -10
        return 1

    def render(self, episode):
        self.draw_gradient_background()
        pygame.draw.rect(self.screen, GREEN, (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y))

        pygame.draw.rect(self.screen, self.agent.color, self.agent.rect, border_radius=8)
        for o in self.obstacles:
            color = RED if o.type == "low" else BLACK
            pygame.draw.rect(self.screen, color, o.rect, border_radius=6)

        # UI
        pygame.draw.rect(self.screen, BLACK, (10, 10, 240, 65), border_radius=8)
        pygame.draw.rect(self.screen, WHITE, (10, 10, 240, 65), 2, border_radius=8)

        score_txt = self.font.render(f"Score: {self.score}", True, WHITE)
        ep_txt = self.font.render(f"Episode: {episode}", True, WHITE)
        self.screen.blit(score_txt, (20, 20))
        self.screen.blit(ep_txt, (20, 50))
        pygame.display.flip()

    def close(self):
        pygame.quit()
        sys.exit()

# ==== MAIN ====
def main():
    global EPSILON
    game = Game()
    total_episodes = 200

    for ep in range(1, total_episodes + 1):
        game.reset()
        episode_reward = 0

        while not game.game_over:
            game.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()

            obs = next((o for o in game.obstacles if o.rect.right >= game.agent.rect.right), None)
            state = get_state(game.agent, obs)
            action = choose_action(state, EPSILON)
            reward = game.update(action)
            next_state = get_state(game.agent, obs)

            old_q = get_Q(state)[action]
            future_q = max(get_Q(next_state))
            Q_table[state][action] = (1 - ALPHA) * old_q + ALPHA * (reward + GAMMA * future_q)

            game.render(EPISODE_OFFSET + ep)
            episode_reward += reward

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        print(f"Episode {EPISODE_OFFSET + ep}: Score = {game.score}, Reward = {episode_reward}")
        print(f"Total states in Q-table: {len(Q_table)}")

        with open("q_table.pkl", "wb") as f:
            pickle.dump(Q_table, f)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"✅ Q-table saved after Episode {EPISODE_OFFSET + ep} at {timestamp}\n")

if __name__ == "__main__":
    main()
