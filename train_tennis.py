from unityagents import UnityEnvironment

from DDPG_agent import MADDPG
import time
import datetime
import torch
from utils import *
from config import Config
import sys

matplotlib.use('tkagg')                             # needed to run on AWS wiht X11 forwarding
sys.stdout = Logger('ddpg_log.log')                 # copies most output to a file
print('************************************************************************************')
cwd = os.getcwd()           # Current Working Directory
if os.name == 'nt':         # 'nt' is what my laptop reports
    cwd = cwd + os.sep + "Tennis_Windows_x86_64"
    env = UnityEnvironment(file_name=cwd + os.sep + "Tennis.exe")
else:                       # assume AWS headless
    cwd = cwd + os.sep + "Tennis_Linux_NoVis"  #omit '_NoVis' to run with graphics,
    env = UnityEnvironment(file_name=cwd + os.sep + "Tennis.x86_64")

# get the default brainasd
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]


def train(env, config):
    agent = MADDPG(state_size=state_size, action_size=action_size, num_agents = num_agents, config=config)
    print_misc(os.path.basename(__file__))  # Prints some basic info for logging

    max_t = 1000
    scores_hist = []
    epsilon = 1
    epsilon_min = .05
    plotter = Plot_Scores()
    step_count = 0
    total_step_count = 0
    for episode in range(config.max_episodes):
        start_t = time.time()  # capture the episode start time  TODO: it would be nice to capture start of train time and report clock duration of solve.
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        agent.reset()
        step_count = 0
        scores = np.zeros(num_agents)  # reset the episode scores tally
        states = env_info.vector_observations  # get the current state (for each agent)
        for t in range(max_t):
            actions = agent.act(states, epsilon)  # get actions (20 in this case)
            env_info = env.step(actions)[brain_name]  # get return from environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            agent.step(states, actions, rewards, next_states, dones)  # save SARSD to buffer and learn from buffered SARSD
            states = next_states  # prep for next pass
            step_count += 1
            total_step_count +=1
            if np.any(dones):  # exit loop if episode finished
                break
        epsilon *= config.epsilon_decay  # explore a wee bit less
        epsilon = max(epsilon, epsilon_min)  # always explore a little
        # episode_t = time.time() - start_t  # wow that took a long time.
        scores_hist.append(np.max(scores))  # keep track of score history

        if episode % 1 == 0 and episode > 2:
            plotter.plot(scores_hist)
            print_rewards(current_scores=scores, scores_hist=scores_hist, steps=step_count, total_steps_count = total_step_count, epsilon=epsilon, ave_len=100)

        if episode % 10 == 0 and episode > 20:  # Lets occasionally save the weights just in case
            agent.save_maddpg()

        if len(scores_hist) > 100 and np.mean(scores_hist[-100:]) >= .5:  # yippee!
            print('Met project requirement in {} episodes'.format(
                episode + 1))  # TODO: we coule probably stop the training here, or only send this message once
    plotter.save()
    return scores_hist


config = Config()
print(config.__dict__)              #record setup to log file
scores_hist = train(env, config)
f = open('scores_hist.txt', "a")
for l in range(len(scores_hist)):
    f.write(scores_hist[l])
f.close()


plt.show()
