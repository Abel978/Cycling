import numpy as np
from copy import deepcopy
from itertools import chain
from helper import custom_flatten as c_flatten

action_mapping = {
    0: 0,           # cost: 0
    1: 1,           # cost: 1
    2: 1.4,         # cost: 1
    3: 2            # cost: 2
}

lazy_indexing = {
    (0,0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (2,0):3
}


class State:
    def __init__(self, p, e) -> None:
        
        if not [len(x) for x in p] == [len(x) for x in e]:
            raise ValueError("Shape of p and e must be the same")
        
        self.p = p
        self.e = e
        self.shape = [len(x) for x in self.p]
    


class OneStepEnv:
    """
    Environment where only a single step is possible,
    i.e. termination after first step
    """
    def __init__(self,
                 S0: State) -> None:
        """
        Args:
            S0: Initial state
        """
        
        self.S0 = S0

        self.state = None
        self.reset()

        self.shape = S0.shape
    
    def reset(self):
        self.state = self.S0
    
    @staticmethod
    def find_indices(element, list_of_lists):
        indices = []
        for i, sublist in enumerate(list_of_lists):
            for j, item in enumerate(sublist):
                if item == element:
                    indices.append((i, j))
        return indices

    def reward(self, S: State) -> np.array:
        """
        payoffs:
            0 for everyone that is not in the lead
            e / sum(e in the leading group) for everyone in the leading group
        """

        """
        Die Auszahlungen stimmen noch nicht. Jeder aus dem Team bekommt die selbe Auszahlung,
        und, wenn zwei Leute vom gleichen Team in der Fuehrungsgruppe sind, zaehlt nur der mit 
        mehr energie. Wenn beide gleich viel haben, werden die trotzdem wie einer behandelt.

        also p = [[2], [2,2], [2]] & e = [[1], [1,1], [0]]
        sollten zu einer auszahlung von 0.5, 0.5, 0.5 und 0 fuehren.
        ebenso wie wenn in der situation e = [[2], [1,2], [0]] waere. (Ob das so sinnvoll ist weiss ich auch nicht)
        """
        

        e_flat = self.custom_flatten(S.e)
        p_flat = self.custom_flatten(S.p)

        leaders = np.array([1 if p == max(p_flat) else 0 for p in p_flat])
        leading_team_index = [max(x) for x in self.custom_reshape(leaders, S.shape)]

        es = [e_flat[i] if leaders[i] == 1 else -1 for i in range(len(e_flat))]
        team_es = [max(x) for x in self.custom_reshape(es, S.shape)]
        # -1 if not leading, else the energy level of the agent who leads

        if sum([x for x in team_es if x >=0]) == 0:
            team_rewards = [1 / sum(leading_team_index) if x == 1 else 0 for x in leading_team_index]
        
        else:
            team_rewards = [e*i/sum([x for x in team_es if x >=0]) for e, i in zip(team_es, leading_team_index)]

        r = []
        for j, l in enumerate(S.shape):
            r = r + [team_rewards[j]] * l
        
        return r

    @staticmethod
    def custom_reshape(arr, shape):
        """
        Reshape array to custom shape, as numpy complains
        "ValueError: cannot reshape array of size 4 into shape (2,1,1)"
        """
        result = []
        index = 0
        for size in shape:
            result.append(list(arr[index:index + size]))
            index += size
        return result
    
    @staticmethod
    def custom_flatten(arr):

        return c_flatten(arr)

    def step(self, a):
        """
        Args:
            a: Action taken by the agents, shape = (num_players, )
                0: Rest
                1: Pace
                1.4: Mark
                2: Attack
        Returns:
            s: Next state
            r: Reward
            d: Done (always true as we have a one-step env)
            i: Info (always {})
        """

        a = self.custom_reshape(np.array([action_mapping[x] for x in a]), self.shape)

        a_flat = self.custom_flatten(a)
        #works for arbitrary number of teams, riders
        #shape = np.shape(a)#team structure of the race

        pos = deepcopy(self.custom_flatten(self.state.p))
        ind = np.unique(pos)
        add = np.zeros(np.shape(pos))
        for j in ind:
            temp = np.zeros(np.shape(pos))
            temp[pos == j] = a_flat[pos==j] #strategies of the current group
            if 2 in temp:
                temp[temp == 1.4] = 2
            else:
                temp[temp == 1.4] = 0 #order matters: the 1.4 guy moves at 1 if there is no 2 but a 1, if there is none of either, he moves at 0!
                
            if 1 in temp:
                temp[temp == 0] = 1 #if there is a helper, all 0's move at speed 1

            temp[pos != j] = 0
            add = add + temp

        pos = pos+add
        pos = self.custom_reshape(pos, self.shape)
        new = self.custom_flatten(self.state.e) - np.round(a_flat) 

        new_state = State(p=pos,e = self.custom_reshape(new, self.shape))

        self.state = new_state

        return new_state, self.reward(new_state), True, {}


if __name__ == "__main__":

    p0 = np.array([[0,0], [0], [2]])
    e0 = np.array([[2,2], [3], [0]])

    S = State(p0, e0) 
    env = OneStepEnv(S)

    a = [0, 1, 2, 3]

    env.step(a)