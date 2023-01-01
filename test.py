from agent import Agent
import gym

agent = Agent(100000, [0, 1], 0, 0, 0.99, 0.00025)
agent.model.load_state_dict(torch.load("best_model.pt"))

env = gym.make("CartPole-v1", render_mode='human')

for episode in range(10):
    state = env.reset()[0]
    score = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        state = next_state