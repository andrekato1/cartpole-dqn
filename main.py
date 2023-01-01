from tqdm import tqdm
from agent import Agent
from tensorboardX import SummaryWriter
import gym

env = gym.make("CartPole-v1")
writer = SummaryWriter()

MEM_SIZE = 100000

agent = Agent(mem_len=MEM_SIZE, possible_actions=[0, 1], initial_eps=1, eps_min=0.05, gamma=0.99, lr=0.0001)

losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
index = 0
episodes = 100000
training_starts = 10000
max_reward = 0

for i in tqdm(range(episodes)):
    obs = env.reset()[0]
    done = False
    losses = 0
    ep_len = 0
    total_reward = 0
    while not done:
        ep_len += 1
        a = agent.get_action(obs)
        next_frame, reward, done, _, _ = env.step(a)
        agent.add_experience(obs, a, reward, next_frame, done)
        obs = next_frame
        total_reward += reward
        index += 1

        if index > training_starts:
            loss = agent.train(batch_size=64)
            losses += loss

    if total_reward > max_reward:
        max_reward = total_reward
        agent.save_model('best_model.pt')

    new_eps = agent.eps - 1/5000 #agent.eps*0.997
    agent.eps = max(agent.eps_min, new_eps)
    agent.timesteps += 1
    agent.update_target()

    writer.add_scalar('Loss', losses, global_step=i)
    writer.add_scalar('Reward', total_reward, global_step=i)
    writer.add_scalar('Epsilon', agent.eps, global_step=i)
    writer.add_scalar('Episode Length', ep_len, global_step=i)

    losses_list.append(losses/ep_len)
    reward_list.append(total_reward)
    episode_len_list.append(ep_len)
    epsilon_list.append(agent.eps)

agent.save_model('latest_model.pt')