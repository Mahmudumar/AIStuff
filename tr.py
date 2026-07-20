import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==============================
# SETTINGS
# ==============================
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
GRID_SIZE = 50
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

MAX_MEMORY = 50_000
BATCH_SIZE = 1000
LR = 0.001
TARGET_UPDATE = 100  # update target network every N games

# ==============================
# COLORS
# ==============================
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# ==============================
# SNAKE
# ==============================
class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)

    def move(self, action):
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
        idx = directions.index(self.direction)

        if action == 1:
            new_dir = directions[(idx + 1) % 4]
        elif action == 2:
            new_dir = directions[(idx - 1) % 4]
        else:
            new_dir = directions[idx]

        self.direction = new_dir
        x, y = self.body[0]
        dx, dy = self.direction
        new_head = (x + dx, y + dy)

        self.body.insert(0, new_head)
        self.body.pop()
        return new_head

    def grow(self):
        self.body.append(self.body[-1])

    def collision(self, point=None):
        if point is None:
            point = self.body[0]

        if (point[0] < 0 or point[0] >= GRID_WIDTH or
            point[1] < 0 or point[1] >= GRID_HEIGHT):
            return True

        if point in self.body[1:]:
            return True

        return False


# ==============================
# GAME
# ==============================
class Game:
    def __init__(self, render=False):
        self.render_enabled = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("AI Snake")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 25)

        self.reset()

    def reset(self):
        self.snake = Snake()
        self.food = self.spawn_food()
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH-1),
                   random.randint(0, GRID_HEIGHT-1))
            if pos not in self.snake.body:
                return pos

    def step(self, action):
        self.frame_iteration += 1
        head = self.snake.move(action)

        reward = 0
        done = False

        if self.snake.collision() or self.frame_iteration > 100 * len(self.snake.body):
            reward = -10
            done = True
            return self.get_state(), reward, done, self.score

        if head == self.food:
            self.snake.grow()
            self.food = self.spawn_food()
            self.score += 1
            reward = 10
        else:
            reward = -0.1

        if self.render_enabled:
            self.draw()

        return self.get_state(), reward, done, self.score

    def draw(self):
        self.screen.fill(BLACK)

        for segment in self.snake.body:
            rect = pygame.Rect(segment[0]*GRID_SIZE,
                               segment[1]*GRID_SIZE,
                               GRID_SIZE-1, GRID_SIZE-1)
            pygame.draw.rect(self.screen, GREEN, rect)

        food_rect = pygame.Rect(self.food[0]*GRID_SIZE,
                                self.food[1]*GRID_SIZE,
                                GRID_SIZE-1, GRID_SIZE-1)
        pygame.draw.rect(self.screen, RED, food_rect)

        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def get_state(self):
        head = self.snake.body[0]
        dir_x, dir_y = self.snake.direction

        point_l = (head[0] - dir_y, head[1] + dir_x)
        point_r = (head[0] + dir_y, head[1] - dir_x)
        point_s = (head[0] + dir_x, head[1] + dir_y)

        state = [
            self.snake.collision(point_s),
            self.snake.collision(point_r),
            self.snake.collision(point_l),

            dir_x == -1,
            dir_x == 1,
            dir_y == -1,
            dir_y == 1,

            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
        ]

        return np.array(state, dtype=int)


# ==============================
# MODEL
# ==============================
class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(11, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)
    
    def save(self, file_name="brain.pt"):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="brain.pt"):
        self.load_state_dict(torch.load(file_name))
        self.eval()  # good practice for inference

# ==============================
# AGENT
# ==============================
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet()
        self.target_model = Linear_QNet()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        self.epsilon = max(0.01, 1.0 - self.n_games / 2000)
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state)
        return torch.argmax(prediction).item()

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return

        mini_sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone()

        with torch.no_grad():
            next_q = self.target_model(next_states)

        for i in range(BATCH_SIZE):
            Q_new = rewards[i]
            if not dones[i]:
                Q_new = rewards[i] + self.gamma * torch.max(next_q[i])
            target[i][actions[i]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()


# ==============================
# TRAINING
# ==============================
def train():
    agent = Agent()
    game = Game(render=False)

    while True:
        state_old = game.get_state()
        action = agent.get_action(state_old)
        state_new, reward, done, score = game.step(action)

        agent.remember(state_old, action, reward, state_new, done)
        agent.train_long_memory()

        if done:
            game.reset()
            agent.n_games += 1

            if agent.n_games % TARGET_UPDATE == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

            print("Game:", agent.n_games, "Score:", score)

            if agent.n_games > 3000:
                torch.save(agent.model.state_dict(), "brain.pt")
                break
            
    torch.save(agent.model.state_dict(), "brain.pt")

def test():
    agent = Agent()
    try:
        agent.model.load("brain.pt")
    except Exception as e:
        print(e)

    game = Game(render=True)

    while True:
        state_old = game.get_state()
        action = agent.get_action(state_old)
        state_new, reward, done, score = game.step(action)

        if done:
            game.reset()
            print("Score:", score)
        if score > 10:
            break

if __name__ == "__main__":
    train()
    test()