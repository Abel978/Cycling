from env import State, OneStepEnv
import numpy as np
from copy import deepcopy

def anneal_eps(i, n, min_eps = 0.05):

    return max(1 - i/n, min_eps)


class Single_State_Agent:

    def __init__(self, state: State, env: OneStepEnv) -> None:
        """
        We pass on a dummy state to properly set up the Q table.
        
        Q = len(states) x len(teams) x len(actions)
        We assume that we just have a single state, i.e. len(states = 1)
        """

        self.state = state      # State of interest
        self.env = env

        self.Q = np.zeros([1, sum(self.state.shape), 4])
    
    def select_action(self, exploration_factor = 0.1) -> int:
        """
        Select action based on epsilon greedy strategy
        """

        eps = np.random.random()

        if eps < exploration_factor:

            # valid actions are also encoded by the initial energies
            valid_actions = [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0]
            ]
            return np.array([np.random.choice(valid_actions[i]) for i in range(sum(self.state.shape))])
        else:
            return np.argmax(self.Q[0,:,:], axis=1)
    
    def update_Q(self, action: int, reward: float, done: bool, gamma: float = 1.0) -> None:
        """
        we assume that the rewards are ordered
        """

        assert len(reward) == sum(self.state.shape), "Maybe I have to assign rewards to each player in the team"

        self.Q[0, [0,1,2,3], action] = reward + gamma * np.max(self.Q[0, action]) * (1 - int(done == True))
    
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
                print(self.Q)