<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=180&section=header&text=Reinforcement%20Learning&fontSize=40&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=From%20dynamic%20programming%20to%20Q-networks&descSize=15&descAlignY=58" width="100%" />

**Reinforcement Learning · Tel Aviv University · 2025**

<img src="https://img.shields.io/badge/%F0%9F%8E%93%20Tel%20Aviv%20University-30363d?style=for-the-badge" />
<img src="https://img.shields.io/badge/Course-0368--7501-30363d?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-30363d?style=for-the-badge&logo=python&logoColor=FFD43B" />
<img src="https://img.shields.io/badge/PyTorch-30363d?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" />
<img src="https://img.shields.io/badge/Gym-30363d?style=for-the-badge&logo=openaigym&logoColor=white" />

</div>

<br/>

> [!NOTE]
> My solutions to the Reinforcement Learning course homework: written answers plus code, from random search to Q-networks.

## 📚 Assignments

| | Assignment | Code |
|:-:|---|---|
| 1️⃣ | **[HW1](HW1/)** | Random search policy for **CartPole** · a PyTorch classifier on **MNIST** · dynamic programming for the most likely word under a letter-level language model |
| 2️⃣ | **[HW2](HW2/)** | **Policy iteration** and **value iteration** |
| 3️⃣ | **[HW3](HW3/)** | Model-based control for **cart-pole** (learned transitions + value iteration) · **tabular Q-learning** and a **Q-network** on FrozenLake |

Written solutions: [HW1](HW1/RL_A1_written.pdf) · [HW2](HW2/RL_A2_written.pdf)

## 🗂️ Structure

```
RL_0368_7501/
├── HW1/
│   ├── cart_pole.py          # random search on CartPole
│   ├── mnist.py              # MNIST classifier
│   ├── language_model.py     # most likely word (dynamic programming)
│   ├── hw1_rl_25.pdf         # assignment
│   └── RL_A1_written.pdf     # my written solution
├── HW2/
│   ├── policy_iter.py        # policy iteration
│   ├── value_iter.py         # value iteration
│   ├── HW2 - 2025.pdf        # assignment
│   └── RL_A2_written.pdf     # my written solution
└── HW3/
    ├── control.py            # model-based cart-pole control
    ├── cart_pole.py          # cart-pole simulator
    ├── tabular_Q.py          # tabular Q-learning (FrozenLake)
    ├── network_Q.py          # Q-network (FrozenLake)
    └── HW 3 - 2025.pdf       # assignment
```

> [!IMPORTANT]
> These are my personal solutions, shared for learning purposes. If you are currently taking this course, please follow its academic integrity guidelines.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%" />
