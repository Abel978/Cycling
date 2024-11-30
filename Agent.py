from env import State, OneStepEnv
import numpy as np
from copy import deepcopy
from helper import custom_flatten

def anneal_eps(i, n, min_eps = 0.00):

    return max(1 - 2 * i/n, min_eps)


class Single_State_Agent:

    def __init__(self, state: State, env: OneStepEnv, alpha=0.3, initial_beliefs: float = 0.0) -> None:
        """
        We pass on a dummy state to properly set up the Q table.
        
        Q = len(states) x len(teams) x len(actions)
        We assume that we just have a single state, i.e. len(states = 1)
        """

        self.state = state      # State of interest
        self.env = env
        self.alpha = alpha

        self.Q = np.zeros([1, sum(self.state.shape), 4]) + initial_beliefs

        self.agent_ind = list(range(sum(self.state.shape)))     # Index of the agent

        self.pre_compute_valid_actions()

    def pre_compute_valid_actions(self):
        """
        Pre-compute valid actions for each player in the team
        """

        self.valid_actions = [list(range(4)) if el > 1 else list(range(2)) if el == 1 else list(range(1)) for el in custom_flatten(self.state.e)]
    
    def select_action(self, exploration_factor = 0.1) -> int:
        """
        Select action based on epsilon greedy strategy
        """

        eps = np.random.random()

        if eps < exploration_factor:
            return np.array([np.random.choice(self.valid_actions[i]) for i in range(sum(self.state.shape))])
        else:
            return np.argmax(self.Q[0,:,:], axis=1)
    
    def update_Q(self, action: int, reward: float, done: bool, gamma: float = 1.0) -> None:
        """
        we assume that the rewards are ordered
        """

        assert len(reward) == sum(self.state.shape), "Maybe I have to assign rewards to each player in the team"

        # the target calculation does not yet work perfect. Doesn't matter however, for now, as we only have one step MDPs anyway
        target = reward 
        #target = reward + gamma * np.max(self.Q[0, action]) * (1 - int(done == True))

        self.Q[0, self.agent_ind, action] += self.alpha * (target - self.Q[0, self.agent_ind, action])
    
    def train(self, 
              num_episodes: int,
              verbose: bool = True):
        
        for m in range(num_episodes):

            temp = deepcopy(self.Q)

            self.env.reset()
            action = self.select_action(exploration_factor=anneal_eps(m, num_episodes))
            _, r, done, _ = self.env.step(action)

            self.update_Q(action, r, done)

            if m % (int(num_episodes/100)) == 0 and verbose:
                print(f"Episode {m} - dQ: {np.max(abs(self.Q - temp))}")
                #print(self.Q)