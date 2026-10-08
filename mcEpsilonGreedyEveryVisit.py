import numpy as np

global state_space, action_space, pits, goals
state_space = {'s1': [1,1], 's2': [1,2],
               's3': [2,1], 's4': [2,2]}
#state_space = {'s1': [1,1], 's2': [1,2], 's3': [1,3], 's4': [1,4], 's5': [1,5],
               #'s6': [2,1], 's7': [2,2], 's8': [2,3], 's9': [2,4], 's10': [2,5],
               #'s11': [3,1], 's12': [3,2], 's13': [3,3], 's14': [3,4], 's15': [3,5],
               #'s16': [4,1], 's17': [4,2], 's18': [4,3], 's19': [4,4], 's20': [4,5],
               #'s21': [5,1], 's22': [5,2], 's23': [5,3], 's24': [5,4], 's25': [5,5]}
action_space = {"↑": 1, "→": 2, "↓": 3, "←": 4, "○": 5}
# 2,1] is the pit in 2x2
# [2,2] is the goal in 2x2
# [2,2], [2,3], [3,3], [4,2], [4,4], [5,2] are pits in 5x5
# [4,3] is the goal in 5x5

#pits = [[2,2], [2,3], [3,3], [4,2], [4,4], [5,2]]
#goals = [[4,3]]
pits = [[1,2]]
goals = [[2,2]]

def reward(state1, action1, state2):
    if state1 == state2:
        if action1 == 5:
            if state2 in pits:
                return -10
            if state2 in goals:
                return 1
            else: 
                return 0
        else:
            return -1
    else:
        if state2 in pits:
            return -10
        elif state2 in goals:
            return 1
        else:
            return 0


def transition(state1, action1):
    top = min(list(state_space.values()), key = lambda t: t[0])[0]
    left = min(list(state_space.values()), key = lambda t: t[1])[1]
    top_left = left
    bottom_left = left
    bottom = max(list(state_space.values()), key = lambda t: t[0])[0]
    right = max(list(state_space.values()), key = lambda t: t[1])[1]
    top_right = right
    bottom_right = right


    state2 = [0, 0]
    if action1 == 5:
        state2 = state1
    else: 
        #up
        if state1[0] == top:
            # stays and cannot go up
            if action1 == 1:
                state2 = state1
            # goes right    
            elif action1 == 2:
                # cannot move right when it is in the [1,5]
                if state1[1] == top_right:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] + 1
            # goes left
            elif action1 == 4:
                # cannot move left when it is in the [1,1]
                if state1[1] == top_left:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] - 1
            # moves down
            else:
                state2[0] = state1[0] + 1
                state2[1] = state1[1]

        #down
        elif state1[0] == bottom:
            # stays and connot move down
            if action1 == 3:
                state2 = state1
            # moves right
            elif action1 == 2:
                # cannot move right when it is in the [5,5]
                if state1[1] == bottom_right:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] + 1
            # moves left
            elif action1 == 4:
                # cannot move left when it is in the [5,1]
                if state1[1] == bottom_left:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] - 1
            # moves up
            else:
                state2[0] = state1[0] - 1
                state2[1] = state1[1]

        #left 
        elif state1[1] == left:
            # stays and connot go left
            if action1 == 4:
                state2 = state1
            # goes up
            elif action1 == 1:
                state2[0] = state1[0] - 1
                state2[1] = state1[1]
            # goes down    
            elif action1 == 3:  
                state2[0] = state1[0] + 1
                state2[1] = state1[1]
            # goes right
            else:
                state2[0] = state1[0] 
                state2[1] = state1[1] + 1

        # right
        elif state1[1] == right:
            # stays and connot go right
            if action1 == 2:
                state2 = state1
            # goes up
            elif action1 == 1:
                state2[0] = state1[0] - 1
                state2[1] = state1[1]
            # goes down    
            elif action1 == 3:  
                state2[0] = state1[0] + 1
                state2[1] = state1[1]
            # goes left
            else:
                state2[0] = state1[0] 
                state2[1] = state1[1] - 1

        # when agent is not in the corners and edges
        else:
            if action1 == 1:
                state2[0] = state1[0] - 1
                state2[1] = state1[1]
            elif action1 == 2:
                state2[0] = state1[0]
                state2[1] = state1[1] + 1
            elif action1 == 3:
                state2[0] = state1[0] + 1
                state2[1] = state1[1]
            else:
                state2[0] = state1[0]
                state2[1] = state1[1] - 1
    return state2, reward(state1, action1, state2)


def selector_policy(state, q, epsilon = 0.5):
    actions = list(action_space.values())
    x = np.random.uniform(0,1)
    if x >= epsilon:
        return max(actions, key= lambda x: q[state][x])
    else:
        #actions_min = [a for a in actions if a != aMax]
        random_action = int(np.random.choice(actions))
        return random_action

def episode_generator(q, epsilon = 0.5):
    num_timesteps = len(state_space) * 10
    episode = []
    state = 's' + str(np.random.randint(1, len(state_space) + 1))
    num_state = state_space[state]
    for t in range(num_timesteps):
        action = selector_policy(state, q, epsilon)
        num_next_state , r = transition(num_state, action)
        episode.append((state, action, r))
        if num_next_state in goals or num_next_state in pits:
            break
        num_state = num_next_state
    return tuple(episode)



def key_of(val):
    for k, v in state_space.items():
        if v == val:
            return k
    return None



def key_of_action(val):
    for k, v in action_space.items():
        if v == val:
            return k
    return None


# ---------------------------------------------------------------- pretty print
def render_box(v_pi, pi, title=None):
    size = int(np.sqrt(len(state_space)))
    """Pretty-print the size x size grid as aligned boxes."""
    W = 11  # cell width

    if title:
        bar = "─" * max(1, 55 - len(title))
        print(f"\n  ── {title} {bar}")

    # 5-column table borders
    top = "  ╔" + ("═" * W + "╦") * (size - 1) + "═" * W + "╗"
    mid = "  ╠" + ("═" * W + "╬") * (size - 1) + "═" * W + "╣"
    bot = "  ╚" + ("═" * W + "╩") * (size - 1) + "═" * W + "╝"

    # action code → arrow (inverse of action_space)
    inv = {v: k for k, v in action_space.items()}

    # special cells (positions are 1-indexed, matching your comments)
    PITS = { tuple(i) for i in pits }
    GOALS = { tuple(i) for i in goals }
    #PITS  = {(2, 2), (2, 3), (3, 3), (4, 2), (4, 4), (5, 2)}
    #GOALS = {(4, 3)}

    def cell(row, col):

        s = f"s{row * size + col + 1}"          # s_1... s_final
        v = float(v_pi[s])
        a = inv[int(pi[s])]
        # small suffix so pits / goal pop out at a glance
        if (row + 1, col + 1) in PITS:
            s = s + "·P"
        elif (row + 1, col + 1) in GOALS:
            s = s + "·G"
        return (
            f"{s:^{W}}",
            f"{('q = ' + format(v, '>7.2f')):^{W}}",
            f"{a:^{W}}",
        )

    grid = [[cell(r, c) for c in range(size)] for r in range(size)]

    print(top)
    for r, row in enumerate(grid):
        for line in range(3):                # name / value / action
            print("  ║" + "║".join(row[c][line] for c in range(size)) + "║")
        if r < size - 1:
            print(mid)
    print(bot)




pi_s = {key : 5 for key in state_space} # initial guess for the policy.

pi_s_prob = {(s, a): 0.0 for s in state_space for a in action_space.keys()} 
q_s = { key1: {key2 : 0.0 for key2 in action_space.values()} for key1 in state_space}
nums = {key1: {key2: 0.0 for key2 in action_space.values()} for key1 in state_space}
returns = {key1: {key2: 0.0 for key2 in action_space.values()} for key1 in state_space}




v_s = {key: 0.0 for key in state_space} #initial guess for the state **optional , just for checking it in the output
gamma = 0.9

print("═" * 60)
print("  Monte Carlo Exploring Starts  •  FrozenLake 5x5  ")
print("═" * 60)
print(f"  Discount factor γ   : {gamma}")
print(f"  Total iterations    : (pending)")

render_box(v_s, pi_s, title=f"Initial guess  (k = {0})")



episodes = []
episodes_num = 30000
A_s_t = len(action_space)
epsilon = 0.5






for k in range(episodes_num):
    episode = episode_generator(q_s, epsilon)
    #all_s_a_pairs = [(s,a) for (s, a, _) in episode]
    #rewards = [r for (s, a, r) inn episode]
    g = 0
    for _ , step in enumerate(episode[::-1]):
        curr_state, curr_action, next_r = step[0], step[1], step[2]
        g = gamma * g + next_r
        returns[curr_state][curr_action] += g
        nums[curr_state][curr_action] += 1
        # policy evaluation
        q_s[curr_state][curr_action] = returns[curr_state][curr_action]/ nums[curr_state][curr_action]
        # policy improvement
        a_max = max(q_s[curr_state], key= q_s[curr_state].get)
        pi_s[curr_state] = a_max
        for a_num in action_space.values():
            if a_num == a_max:
                pi_s_prob[(curr_state, key_of_action(a_num))] = round(1 - epsilon * (A_s_t - 1) / A_s_t, 3)
            else:
                pi_s_prob[(curr_state, key_of_action(a_num))] = round(epsilon / A_s_t, 3)

    v_s[curr_state] = q_s[curr_state][a_max]
    if k % 1000 == 0:
        render_box(v_s, pi_s, title=f"After episode number:(k = {k})")

        

print()
print("═" * 60)
print(f"  Converged after {k} MC Exploring starts steps")
print("═" * 60)
render_box(v_s, pi_s, title="Final optimal policy & state values")
print()

print("probability of every policy")
print(pi_s_prob)
