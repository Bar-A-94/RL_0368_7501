##################################
# Create env
import gym
env = gym.make('FrozenLake-v0')
env = env.env
print(env.__doc__)
print("")

#################################
# Some basic imports and setup
# Let's look at what a random episode looks like.

import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt
import random
#%matplotlib inline
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
# env.seed(0); from gym.spaces import prng; prng.seed(10)

# ==== Changed to make this run! ====
# 1) Seed Python and NumPy if you ever use them
random.seed(10)
np.random.seed(10)

# 2) Seed the env’s internal RNG (dynamics + rewards)
env.seed(0)

# 3) Seed the action‐sampling RNG
env.action_space.seed(10)
# ==== Changed ends ====

# Generate the episode
env.reset()
for t in range(100):
    env.render()
    a = env.action_space.sample()
    ob, rew, done, _ = env.step(a)
    if done:
        break
assert done
env.render();

#################################
# Create MDP for our env
# We extract the relevant information from the gym Env into the MDP class below.
# The `env` object won't be used any further, we'll just use the `mdp` object.

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)
GAMMA = 0.95 # we'll be using this same value in subsequent problems

print("")
print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
for i in range(4):
    print("P[5][%i] =" % i, mdp.P[5][i])
print("")

#################################
# Programing Question No. 2, part 1 - implement where required.

def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    # 1) Build P_pi and R_pi
    P_pi = np.zeros((mdp.nS, mdp.nS))
    R_pi = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        a = pi[s]
        for (probability, nextstate, reward) in mdp.P[s][a]:
            P_pi[s, nextstate] += probability
            R_pi[s]   += probability * reward

    # 2) Form A = I - gamma * P_pi
    I = np.eye(mdp.nS)
    A_mat = I - gamma * P_pi

    # 3) Solve A_mat @ V = R_pi
    V_pi = np.linalg.solve(A_mat, R_pi)
    return V_pi

actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
print("Policy Value: ", actual_val)

#################################
# Programing Question No. 2, part 2 - implement where required.

def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA])
    for state in range(mdp.nS):
        for action in range(mdp.nA):
            for (probability, nextstate, reward) in mdp.P[state][action]:
                Qpi[state, action] += probability * (reward + gamma * vpi[nextstate])
    return Qpi

Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Policy Action Value: ", actual_val)

#################################
# Programing Question No. 2, part 3 - implement where required.
# Policy iteration

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):
        # YOUR CODE HERE
        # you need to compute qpi which is the state-action values for current pi
        
        # Compute the state-value function Vπn
        vpi = compute_vpi(pi_prev, mdp, gamma)

        # Using Vπn , compute the state-action-value function Qπn
        qpi = compute_qpi(vpi, mdp, gamma)

        # Compute new policy π n+1(s) = argmax Qπ n (s, a)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi

    # plot state value over iterations
    x = [i for i in range(1, nIt + 1)]
    plt.figure(figsize=(3, 3))
    for state in range(mdp.nS):
        y = [Vs[it][state] for it in range(nIt)]
        plt.plot(x, y, label=f'state {state}')
    plt.xlabel('Iterations')
    plt.ylabel('State value')
    plt.title(f'Value function per state over {nIt} iterations')
    plt.xticks(x)
    plt.legend(bbox_to_anchor=(1.13, 1.015), loc=1)
    plt.grid()
    plt.show()
    return Vs, pis


Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
plt.plot(Vs_PI);
