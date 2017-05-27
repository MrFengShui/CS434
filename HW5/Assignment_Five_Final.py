import numpy as np
import io

def func_load_file(name):
    m, n = 0, 0
    reward = None
    with open(name, 'r') as file:
        lines = file.readlines()
        tmp = lines[0][:-1].split(' ')
        n, m = int(tmp[0]), int(tmp[1])
        tmp = [float(item) for item in lines[-1][:-1].split()]
        reward = np.array(tmp)
        tmp, count, dataset = lines[1:-1], 0, []
        while True:
            while True:
                key = tmp.pop(0)
                if key == '\n': count += 1
                if count == 1 and key == '\n': tmp_set = []
                if count == 2: break
                if key != '\n': tmp_set.append([float(item) for item in key[:-1].split()])
            dataset.append(np.array(tmp_set))
            if tmp: tmp.insert(0, '\n');count = 0
            else: break
    return n, m, reward, dataset

if __name__ == '__main__':
    print '+++', func_load_file('sample_data.txt')

# https://github.com/mqtlam/osu-cs533/tree/master/assignment3
