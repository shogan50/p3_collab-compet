import numpy as np
import random
import copy
from collections import namedtuple, deque
from DDPG_Model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

class MADDPG():
    def __init__(self, state_size, action_size, num_agents, config):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.config = config
        self.memory = ReplayBuffer(action_size=action_size, config = config)
        self.MADDPG_agents = [
            DDPG_Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, config=config),
            DDPG_Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, config=config)
            ]
        self.f_action_dim = self.action_size*self.num_agents
        
    def reset(self):
        for agent in self.MADDPG_agents:
            agent.reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        config = self.config
        f_states = np.reshape(states,newshape=(-1))
        f_next_states = np.reshape(next_states, newshape=(-1))
        # print('sizes should be (48,) (2, 24) -->>', f_states.shape, states.shape)
        self.memory.add(f_states, states, actions, rewards, f_next_states, next_states, dones)
        
        if len(self.memory) > max(config.batch_size, 100*14):  # there are typically 14 steps in an episode in the beginning.
            for _ in range(config.num_repeats):
                for agent_no in range(self.num_agents):
                    samples = self.memory.sample()
                    self.learn(samples, agent_no)
                self.soft_update_all()
    
    def soft_update_all(self):
        for agent in self.MADDPG_agents:
            agent.soft_update_all()
    
    def learn(self, samples, agent_no):
        f_states, states, actions, rewards, f_next_states, next_states, dones = samples
        # print(f_states.shape, states.shape)
        critic_f_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=self.config.device)
        # print('cfna init', critic_f_next_actions.shape)
        for agent_id, agent in enumerate(self.MADDPG_agents):
            agent_next_state = next_states[:,agent_id,:] # next_states[:,agent_id,:]
            # print('agent_next_state',agent_next_state.shape)
            critic_f_next_actions[:,agent_id,:] = agent.actor_target.forward(agent_next_state)
        # print('cfna', critic_f_next_actions.shape)
        critic_f_next_actions = critic_f_next_actions.view(-1,self.f_action_dim)
        # print('cfna',critic_f_next_actions.shape)
        agent = self.MADDPG_agents[agent_no]
        agent_state = states[:,agent_no,:]
        actor_f_actions= actions.clone()
        actor_f_actions[:,agent_no,:] = agent.actor_local.forward(agent_state)
        actor_f_actions = actor_f_actions.view(-1, self.f_action_dim)
        
        f_actions           = actions.view(-1, self.f_action_dim)
        agent_rewards       = rewards[:,agent_no].view(-1,1)
        agent_dones         = dones[:,agent_no].view(-1,1)
        experiences = (f_states, actor_f_actions, f_actions, agent_rewards, agent_dones, f_next_states, critic_f_next_actions)
        agent.learn(experiences,gamma = self.config.gamma)
        
    def act(self, f_states, i_episode, add_noise=True):
        # print(f_states)  # (2x24)
        # print(np.reshape(f_states[0,:],newshape=(-1,1))) # (24x1)
        actions = []
        for agent_id, agent in enumerate(self.MADDPG_agents):
            # action = agent.act(np.reshape(f_states[agent_id,:],newshape=(-1,1)), i_episode, add_noise)
            action = agent.act(np.reshape(f_states[agent_id,:], newshape=(1,-1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))
            actions.append(action)
        actions = np.concatenate(actions, axis=0)

        return actions
        
    def save_maddpg(self):
        for agent_id, agent in enumerate(self.MADDPG_agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local_'    + str(agent_id) + '.pth')
            torch.save(agent.critic_local.state_dict(),'checkpoint_critic_local_'   + str(agent_id) + '.pth')
            
    def load_maddpg(self):
        pass
                    

class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            config Config(): config class

        """
        self.config = config
        device = config.device
        self.tau = config.tau
        self.state_size = state_size
        self.action_size = action_size
        random_seed = config.seed

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size,
                                 action_size,
                                 random_seed,
                                 fc1_units=config.fc1_units,
                                 fc2_units=config.fc2_units
                                 ).to(device)
        self.actor_target = Actor(state_size,
                                  action_size,
                                  random_seed,
                                  fc1_units=config.fc1_units,
                                  fc2_units=config.fc2_units
                                  ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * num_agents,
                                   action_size * num_agents,
                                   random_seed,
                                   fcs1_units=config.fc1_units, 
                                   fc2_units=config.fc2_units
                                   ).to(device)
        self.critic_target = Critic(state_size * num_agents,
                                    action_size * num_agents,
                                    random_seed, 
                                    fcs1_units=config.fc1_units,
                                    fc2_units=config.fc2_units
                                    ).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_critic, weight_decay=config.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, sigma=config.sigma)

        # Replay memory
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

    def act(self, states, add_noise=True, epsilon=1):
        """Returns actions for given state as per current policy."""
        actions =[]
        device = self.config.device

        state = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()                                         #set the mode to eval
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()                                        #set the mode back to train
        if add_noise:
            actions += self.noise.sample() * epsilon
            # actions +=  0.5*np.random.randn(1,self.action_size)
        return np.clip(actions, -1, 1)                                  #the noise can bring this out of range of -1,1


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):

        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        f_states, actor_f_actions, f_actions, agent_rewards, agent_dones, f_next_states,\
            critic_f_next_actions = experiences  #f_ means full
        # ---------------------------- update critic ---------------------------- #
        Q_targets_next = self.critic_target(f_next_states, critic_f_next_actions) # Q'(si+1,µ'(si+1|θµ')|θQ')  https://arxiv.org/pdf/1509.02971.pdf Algorithm 1
        # Compute Q targets for current states (y_i)
        Q_targets = agent_rewards + (gamma * Q_targets_next * (1 - agent_dones))
        # Compute critic loss
        Q_expected = self.critic_local(f_states, f_actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local(f_states, actor_f_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    #
    def soft_update_all(self):
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model,tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, target, source):   # couldn't you just do a soft update with Tau = 1.0????
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)   

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=config.buffer_size) # internal memory (deque)
        self.batch_size = config.batch_size
        self.experience = namedtuple("Experience", field_names=["f_state",
                                                                "state",
                                                                "action",
                                                                "reward",
                                                                "f_next_state",
                                                                "next_state",
                                                                "done"])
        self.seed = random.seed(config.seed)
        self.config = config

    # self.memory.add(f_states, states, actions, rewards, f_next_states, dones)
    def add(self, f_state, state, action, reward, f_next_state, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(f_state, state, action, reward, f_next_state, next_state, done)
        self.memory.append(e)

    def sample(self):
        device = self.config.device
        # probs = []
        # mu = .75
        # for idx in range(len(self.memory)):
        #     e = np.array(self.memory[idx]['reward'])
        #     if np.sum(e)==0:
        #         probs.append(mu)
        #     else:
        #         probs.append(1/mu)
        # probs = np.array(probs)
        # probs = probs/np.sum(probs)


        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size, probs=probs)
        f_states    = torch.from_numpy(np.array([e.f_state for e in experiences if e is not None]))\
            .float().to(device)
        states      = torch.from_numpy(np.array([e.state for e in experiences if e is not None]))\
            .float().to(device)
        actions     = torch.from_numpy(np.array([e.action for e in experiences if e is not None]))\
            .float().to(device)
        rewards     = torch.from_numpy(np.array([e.reward for e in experiences if e is not None]))\
            .float().to(device)
        f_next_states = torch.from_numpy(np.array([e.f_next_state for e in experiences if e is not None]))\
            .float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None]))\
            .float().to(device)
        dones       = torch.from_numpy(np.array([e.done for e in experiences if e is not None])
            .astype(np.uint8)).float().to(device)
        # print('sample', f_states.shape, states.shape)
        return f_states, states, actions, rewards, f_next_states, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)