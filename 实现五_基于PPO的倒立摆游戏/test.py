import gymnasium as gym
import torch
from model import PolicyNet


def test_render(agent, env, episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"回合 {episode + 1}: 总奖励 = {total_reward}")
    env.close()


# 构建模型并加载权重
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
state_dim = 4
action_dim = 2
hidden_dim = 128

actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
actor.load_state_dict(torch.load('policy_model.pth', map_location=device))
actor.eval()


class AgentWrapper:
    """包装一下，让 test_render 可以复用"""
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


agent_wrapper = AgentWrapper(actor, device)

# 加载模型后测试
env = gym.make('CartPole-v1', render_mode='human')
test_render(agent_wrapper, env)
