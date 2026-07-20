from collections import defaultdict
import numpy as np

action_idx = 0
state = (4, 4, (0,-1))
q_table = defaultdict(lambda: np.zeros(4))

# q_table[state][action_idx] += .1 * (2 - 20)

idx = np.argmax(q_table[state])
# print(idx)

# Default dict is a generator of values for any lookup based on the function
# specified in the definition


# a =defaultdict(list)



print(q_table)