import gymnasium as gym
from model import Agent
import torch


def test_render(agent, env, episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action, _ = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"回合 {episode + 1}: 总奖励 = {total_reward}")
    env.close()


# 加载模型后测试
env = gym.make('CartPole-v1', render_mode='human')
agent = Agent()
agent.pi.load_state_dict(torch.load('policy_model.pth'))
agent.pi.eval()
test_render(agent, env)
