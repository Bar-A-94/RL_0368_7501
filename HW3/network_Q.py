import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.autograd import Variable


# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# My code:
net = nn.Sequential(nn.Linear(16, 4, bias=False))
criterion = nn.MSELoss(reduction='sum')
optim = torch.optim.SGD(net.parameters(), lr=0.1)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # My code:
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        state = torch.zeros(1,16)
        state[0, s] = 1.0
        Q_pred = net(state)          # shape [1,4]

        a = torch.argmax(Q_pred, dim=1).item()

        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        with torch.no_grad():
            next_state = torch.zeros(1,16)
            next_state[0, s1] = 1.0
            Q1 = net(next_state)
            # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
            maxQ1 = torch.max(Q1)
            Q_target = Q_pred.clone()
            Q_target[0, a] = r + y * maxQ1


        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        optim.zero_grad()
        step_loss = criterion(Q_pred, Q_target)
        step_loss.backward()
        optim.step()

        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
