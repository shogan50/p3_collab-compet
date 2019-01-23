[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

This work was turned in for credit for project 3 of the Udacity Deep Reinforcement Nano Degree.  The objective was to train an agent, two agents in this case, to collaborate/compete in a game similar to tennis in 2d. A modified version of the Unit ML-Agents [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment was provided.  

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

This code in this repo was executed on an AWS p2.xlarge instance with a headless version of the environment and x11 fowarding for graphs.  If running on a linux platform, it should run as is.  If it is desired to run on linux and see the graphics, the environment is included in the repo.  It is necessary only to remove the '_NoVis' from the path of the agent. `cwd = cwd + os.sep + "Tennis_Linux_NoVis"` -> `cwd = cwd + os.sep + "Tennis_Linux"` in train_tennis.py.

This project started on a windows platform, but the anti virus update installed on the company laptop I use began preventing the environment from running.  It may work out of the box on windows if the line `matplotlib.use('tkagg')` is omitted from train_tenns.py.


1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)

    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the unzipped folder in the `p3_collab-compet/` folder.  If on linux, it will be necessary to set execute permissions.

### Instructions

From the command line, `execute train_tennis.py`. 
