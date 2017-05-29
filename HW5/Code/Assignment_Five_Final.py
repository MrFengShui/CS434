import numpy as np
import io

np.set_printoptions(precision=4)

def func_load_file(name):
    m, n = 0, 0
    with open(name, 'r') as file:
        lines = file.readlines()
        tmp = lines[0][:-1].split(' ')
        n, m = int(tmp[0]), int(tmp[1])
        tmp = [float(item) for item in lines[-1][:-1].split()]
        reward = np.array(tmp)
        tmp, count, matrix = lines[1:-1], 0, []
        while True:
            while True:
                key = tmp.pop(0)
                if key == '\n': count += 1
                if count == 1 and key == '\n': tmp_set = []
                if count == 2: break
                if key != '\n': tmp_set.append([float(item) for item in key[:-1].split()])
            matrix.append(np.array(tmp_set))
            if tmp: tmp.insert(0, '\n');count = 0
            else: break
    return n, m, reward, matrix

def func_horizon_eval(mdp, vec, beta):
    n = mdp.get_num_state()
    I, T = np.eye(n), np.zeros((n, n))
    for state_idx in range(n):
        T[state_idx, :] = mdp.get_trans_prob(state_idx, vec)
    return np.dot(np.linalg.inv(I - beta * T), mdp.get_reward())

def func_value_iter(mdp, beta, epsilon = 0.001):
    curr_vec, vec = np.zeros((mdp.num_state,)), np.zeros((mdp.num_state,), dtype = int)
    prev_vec = np.array(curr_vec)
    for i in range(100000):
        for state_idx in range(mdp.num_state):
            expects = [np.dot(mdp.get_trans_prob(action_idx, state_idx), prev_vec) for action_idx in range(mdp.num_action)]
            curr_vec[state_idx] = mdp.get_reward(state_idx) + beta * np.max(expects)
        if np.linalg.norm(prev_vec - curr_vec, ord = np.inf) < epsilon: break
        prev_vec = np.array(curr_vec)
    for state_idx in range(mdp.num_state):
        expects = [np.dot(mdp.get_trans_prob(action_idx, state_idx), curr_vec) for action_idx in range(mdp.num_action)]
        vec[state_idx] = np.argmax(expects)
    return curr_vec, vec

class MDP:

    def __init__(self, name):
        self.num_state, self.num_action, self.reward, self.matrix = func_load_file(name)

    def get_trans_prob(self, action_idx, state_idx, next_state = None):
        return self.matrix[action_idx][state_idx, :] if next_state == None else self.matrix[action_idx][state_idx, next_state]

    def get_reward(self, state_idx = None):
        return self.reward if state_idx == None else self.reward[state_idx]

if __name__ == '__main__':
    mdp = MDP('MDP1.txt')
    for beta in np.arange(0.1, 1.0, 0.1):
        vec, proc = func_value_iter(mdp, beta)
        print 'Beta:', beta, 'Vector:', vec, 'Process:', proc

# https://github.com/mqtlam/osu-cs533/tree/master/assignment3
