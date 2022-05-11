# Report: Description of the implementation.

The Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm below successfully solves the task of this project. It learns from the provided environment without any prior knowledge of it or data labels and maximizes reward by interacting with the environment.

## Algorithm structure
The maddpg.py file contains a MultiAgent class which utilizes the Agent class of ddpg.py. The DDPG agents call an Actor and a Critic Class from model.py, as usual, which also contains their pytorch instructions to define the NN architecture.

## Learning algorithm summary

This algorithm trains two separate agents to take actions based on their own observations as well as centralized critics with additional information of all agents. This algorithms builds on the concept of DDPG and brings it to multi-agent tasks by separating out their observations and thereby avoiding the apparent non-stationarity from the perspective of any individual agent in multi-agent environments. This approach does not contain the shortcoming of policy gradient algorithms, like REINFORCE, which exhibit high variance gradient estimates that are cumulating exponentially with the number of agents (leading to exponential decrease of the probability of taking a gradient step in the right direction), and render them unsuitable for problems with a large number of agents.

## Learning algorithm details
The official paper can be found here: https://arxiv.org/pdf/1706.02275.pdf and the official repo can be found here: https://github.com/openai/maddpg. The paper is exploring deep RL methods for multi-agent domains and introduces MADDPG to solve (among other challenges) the non-stationarity of single agent's environments.

The MADDPG in this repo is an off-policy multi-agent actor-critic approach that uses the concept of target networks with centralized training and decentralized execution. In the reference paper it is used for mixed cooperative-competitive environments. 

If we dissect these terms, this means:
- Off-policy means that the agent updates its network using the expected return assuming a greedy policy will be followed (while on-policy approaches estimate the value of a policy while using it for control)
- Multi-agent means that we are working with a system with more than one agent (in this case 2). This leads to interactions between agents, introducing more complexity and needs for novel approaches
- Actor-critic means that the algorithm trains two networks at the same time for function approximation, where the actor learns the policy function mu which returns the optimal action(s) and the critic learns the value function to evaluate the actor's actions and helps improve the training ability
- Target networks means the idea of creating a local and a target network to break the correlation between the target from the actions and thereby stabilize the learning
- Mixed cooperative-competitive means a mixture of cooperative and competitive environments. In cooperative environments, the agents are only concerned about a group task with an across-agent reward. In a competitive environment, each agents is only concered about their own respective reward, where one agent's loss could be the other agent's gain (e.g., a game of two agents playing soccer against each other).
- Centralized training and decentralized execution means that extra information is used by the critic compared to the actor, like states observed and agents taken by other agents. The actors only have access to their own observations and actions:
<img width="431" alt="Bildschirmfoto 2022-05-11 um 19 46 11" src="https://user-images.githubusercontent.com/23191357/167913599-e71ae284-c195-4135-ab3e-ed52d0a38f02.png">


In general, the approach considers a game with N agents with policies parametrized by Theta_1 to Theta_N and policies pi_1 to pi_N for all agents. The gradient of the expected return for agent i, is then, given...

- simple gradient update, which directly adjusts the policy parameters theta to maximize the objective function J
- where state s is assumed via greedy policy mu and the actions a_i come from policy pi_i,
- and Q_pi_i = Q_mu_i being the centralized action-value function, as explained above, that helps to consider extra information from other agents
- and an experience replay buffer D
- and working with N deterministic continuous policies mu_theta_i
the gradient can be formulated as 

<img width="1055" alt="Bildschirmfoto 2022-05-11 um 20 03 05" src="https://user-images.githubusercontent.com/23191357/167916412-0a8742eb-cc63-41c9-b7bf-70147809afbb.png">

with the centralized action-value function being updated as 

<img width="1126" alt="Bildschirmfoto 2022-05-11 um 20 05 50" src="https://user-images.githubusercontent.com/23191357/167916837-1a6655bf-03a9-48b2-ad7b-30e3b55946d8.png">

## Technical implementation details

The algorithm trains two agents with exactly the same Neural network architectures:
- Actor
  - Input layer (size 8)
  - FC layer (size 256)
  - Relu
  - FC layer (size 128)
  - Relu
  - Output layer (size 2)
  - Tanh
- Critic
  - Input layer (size 8)
  - FC layer (size 258)
  - Relu
  - FC layer (size 128)
  - Relu
  - Output layer (size 1)

The DDPG agent contains the following components and configs:
- A replay buffer to store memories with the size of 1e5
- Minibatch sizes of 256
- A discount factor of 0.99 for value function approximation
- A soft update to blend the regular into the target network of 1e-3
- Learning rates of the actor and critic each set to 1e-4
- Noise according to the Ornstein-Uhlenbeck process with theta=0.15, sigma=0.2
- Repetitions of learning per agent-step of 3

## Reaching of Rewards

The jupyetr notebook shows that the agent was able to receive an average reward of +0.5 over 100 episodes after 2,745 episodes. In the previous trainig run, it took 2,241 episodes. Interesting is that this training contained more instabilities between episodes 2,300 and 2,700, whereas the agents always managed to recover from the local minima and gradients in the wrong direction.

<img width="717" alt="Bildschirmfoto 2022-05-11 um 22 59 50" src="https://user-images.githubusercontent.com/23191357/167946685-69f7d3f3-db1a-445a-bae7-0f221120c199.png">

## Ideas for Future Work

Future ideas for improving the agent's performance:
- Read the literature and test implementations of other modern multi-agent approaches, e.g.
  - MAA2C
  - IQL
  - IDDPG
  - IPPO
- Tune hyperparameters to accelerate training and better understand how the agent can start working more quickly (not only after 2k+ episodes) 

