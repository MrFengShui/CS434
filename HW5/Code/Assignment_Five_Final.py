import numpy as np
import io

np.set_printoptions(precision=10)

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

def func_horizon_eval(mdp, policy, beta):
    I, matrix = np.eye(mdp.num_state), np.zeros((mdp.num_state, mdp.num_state))
    for state_idx in range(mdp.num_state):
        matrix[state_idx, :] = mdp.get_trans_prob(policy[state_idx], state_idx)
    return np.dot(np.linalg.inv(I - beta * matrix), mdp.get_reward())

def func_policy_opt(mdp, beta):
    policy = np.random.randint(0, mdp.num_action, (mdp.num_state,))
    while True:
        value = func_horizon_eval(mdp, policy, beta)
        tmp_policy = func_calc_policy(value, mdp)
        if np.all(policy == tmp_policy): break
        policy = np.array(tmp_policy)
    return value, policy

def func_calc_policy(value, mdp):
    policy = np.zeros(mdp.num_state, dtype = int)
    for state_idx in range(mdp.num_state):
        expects = [np.dot(mdp.get_trans_prob(action_idx, state_idx), value) for action_idx in range(mdp.num_action)]
        policy[state_idx] = np.argmax(expects)
    return policy

def func_value_iter(mdp, beta, epsilon = 0.0000001):
    value, policy = np.zeros((mdp.num_state,)), np.zeros((mdp.num_state,), dtype = int)
    prev_value = np.array(value)
    for i in range(1000):
        for state_idx in range(mdp.num_state):
            expects = [np.dot(mdp.get_trans_prob(action_idx, state_idx), prev_value) for action_idx in range(mdp.num_action)]
            value[state_idx] = mdp.get_reward(state_idx) + beta * np.max(expects)
        if np.linalg.norm(prev_value - value, ord = np.inf) < epsilon: break
        prev_value = np.array(value)
    policy = func_calc_policy(value, mdp)
    return value, policy

class MDP:

    def __init__(self, name):
        self.num_state, self.num_action, self.reward, self.matrix = func_load_file(name)

    def get_trans_prob(self, action_idx, state_idx, next_state = None):
        return self.matrix[action_idx][state_idx, :] if next_state == None else self.matrix[action_idx][state_idx, next_state]

    def get_reward(self, state_idx = None):
        return self.reward if state_idx == None else self.reward[state_idx]

if __name__ == '__main__':
    mdp = MDP('test-data-for-MDP-1.txt')
    print '========== MDP Value Iteration =========='
    value, proc = func_value_iter(mdp, 0.1)
    print 'Beta:', beta, 'Value:', value, 'Process:', proc
    value, proc = func_value_iter(mdp, 0.9)
    print 'Beta:', beta, 'Value:', value, 'Process:', proc
    # for beta in np.arange(0.1, 1.0, 0.1):
    #     value, proc = func_value_iter(mdp, beta)
    #     print 'Beta:', beta, 'Value:', value, 'Process:', proc
    print '========== MDP Policy Iteration =========='
    value, proc = func_policy_opt(mdp, 0.1)
    print 'Beta:', beta, 'Value:', value, 'Process:', proc
    value, proc = func_policy_opt(mdp, 0.9)
    print 'Beta:', beta, 'Value:', value, 'Process:', proc
    # for beta in np.arange(0.1, 1.0, 0.1):
    #     value, proc = func_policy_opt(mdp, beta)
    #     print 'Beta:', beta, 'Value:', value, 'Process:', proc
