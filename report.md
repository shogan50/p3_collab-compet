# Project 3 - Collaboration - Competition

## Introduction
This project used the MADDPG algorithm.  I tried both DDPG and PPO from previous projects, but as described in this [paper](https://arxiv.org/pdf/1706.02275.pdf), even though the same network is driving both agents, from the perspective of each agent, the environment is not static and therefore convergence is more challenging.  

## Algorithm
Multi Agent Deep Deterministic Policy Gradients or DDPG was used to solve this environment.  MADDPG differs from DDPG in that there are distinct networks representing each agent.  Further, each agent is able to see the entire observation space including the actions of opponents.  

This environment has elements of both collaboration and competition but is largely collaborative as each agent is rewarded only for successfully hitting the ball and benefits from hitting the ball in such a way that the opponent can easily return it.  For fun, I pondered altering the rewards each agent sees such that each agent is actually rewarded for beating the other agent.  

The MADDPG class mostly pre-processes the data so that the DDPG class, one instance for each agent, sees the correct data.  The DDPG class is almost unchanged from project 2 with one interesting exception.  The OUNoise class has a bug to this day, which propagated to this project.  There is a line which should use the `random.randn` function, but instead used the `random.random` method.  This makes all of the returns positive and an unstable positive feedback loop.  It's a small wonder the robot arms ever trained at all.  In fact I tried to use that same code on the crawler without success.  I intend to go back and try again with a proper noise function.  It would also be interesting to see if vanilla DDPG with proper noise function would have solved this project.

## Hyperparameters

Hypterparameter|value
---|---
buffer_size|50000      
batch_size|256           
gamma|0.99              
tau|.02             
LR_actor|1e-3         
LR_critic|3e-3          
weight_decay|0.0        
max_episodes|2500
epsilon_decay|.2**(1/(1500)) (~.998)
fc1_units|256
fc2_units|128
sigma|0.2       
num_repeats|1

## Results
![Training](https://github.com/shogan50/p3_collab-compet/blob/master/misc/plot_image%20trial0.jpeg?raw=true"training")
The project met the requirements at about 900 episodes. It is interesting that at 1700 episodes, it did more poorly.  Something else that I noticed repeatedly was that after a particularly good run, say a score of 2.5, the agent would then miss the ball entirely.  As a test, after this, I hid the most recent 256 experiences in the replay buffer from the agents.  On a sample size of N=1, the agents failed to learn nearly as well.  This is almost useless information however, as I found that the success is highly dependent on how many successes the agent has in the initial episodes before training.  If there were many, the agent generally trained much faster.  Frequently there would be none in the first 150 episodes. 

## Future improvements
1. MAPPO
2. Parameter noise
3. Prioritized Replay

I've not read any papers on MAPPO, but given the fact that rewards come at most every 15 steps, my intuition, such as it is, tells me that PPO with a rollout of at least 15 should work well. (I understand that value still gets learned in the absence of training on a chain of events with a return.)  


There is a paper on adding noise to the network parameters rather than actions which I can't find as I write this.  From memory it resulted in remarkably smoother control of a car in a driving game with simultaneously faster learning.

Given that much isn't going on in 14 or so of every 15 steps, it would be interesting to see what happens if replay was prioritized on transitions with a non-zero score.  I realize this is different than customary prioritized replay.

