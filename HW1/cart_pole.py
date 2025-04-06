import gymnasium as gym
import numpy as np
from tqdm import tqdm
# env = gym.make('CartPole-v1', render_mode='human')
def evaluate_episode(env, agent, weights=None):
    '''Evaluate an episode using the given agent and weights'''
    observation, info = env.reset()
    total_reward = 0

    episode_over = False
    while not episode_over:
        action = agent(env, observation, weights)  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        episode_over = terminated or truncated
    return total_reward

def random_action(env, observation=None, weights=None):
    '''Choose a random action from the action space - used for testing'''
    return env.action_space.sample()


def weighted_action(env, observation, weights):
    action = np.dot(observation, weights)
    return int(action.item() >= 0)
    
def random_search(env):
    best_reward = 0
    best_weights = None
    best_index = None
    for i in range(10000):
        weights = np.random.uniform(-1, 1, size=(env.observation_space.shape[0],))
        total_reward = evaluate_episode(env, weighted_action, weights)
        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = weights
            best_index = i
        if best_reward >= 200:
            return best_weights, best_reward, best_index
    return best_weights, best_reward, 10000

def evaluate_random_search(env):
    number_of_episodes = np.zeros(1000)
    for i in tqdm(range(1000), desc="Random Search Trials"):
        result = random_search(env)
        number_of_episodes[i] = result[2]
    
    print(f"Average episodes needed: {np.average(number_of_episodes):.2f}")
    # Create histogram using matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(number_of_episodes, bins=50, edgecolor='black')
    plt.title('Distribution of Episodes Needed to Find Successful Policy')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('random_search_distribution.png')


env = gym.make('CartPole-v1')
weights = np.random.uniform(-1, 1, size=(env.observation_space.shape[0], 1))
evaluate_random_search(env)
env.close()