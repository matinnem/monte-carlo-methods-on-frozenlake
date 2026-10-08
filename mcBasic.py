import numpy as np

state_space = {'s1': [1,1], 's2': [1,2], 's3': [1,3], 's4': [1,4], 's5': [1,5],
               's6': [2,1], 's7': [2,2], 's8': [2,3], 's9': [2,4], 's10': [2,5],
               's11': [3,1], 's12': [3,2], 's13': [3,3], 's14': [3,4], 's15': [3,5],
               's16': [4,1], 's17': [4,2], 's18': [4,3], 's19': [4,4], 's20': [4,5],
               's21': [5,1], 's22': [5,2], 's23': [5,3], 's24': [5,4], 's25': [5,5]}
action_space = {"↑": 1, "→": 2, "↓": 3, "←": 4, "○": 5}
# [2,2], [2,3], [3,3], [4,2], [4,4], [5,2] are pits
# [4,3] is the goal
global pits, goals
pits = [[2,2], [2,3], [3,3], [4,2], [4,4], [5,2]]
goals = [[4,3]]

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
    state2 = [0, 0]
    if action1 == 5:
        state2 = state1
    else: 
        #up
        if state1[0] == 1:
            # stays and cannot go up
            if action1 == 1:
                state2 = state1
            # goes right    
            elif action1 == 2:
                # cannot move right when it is in the [1,5]
                if state1[1] == 5:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] + 1
            # goes left
            elif action1 == 4:
                # cannot move left when it is in the [1,1]
                if state1[1] == 1:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] - 1
            # moves down
            else:
                state2[0] = state1[0] + 1
                state2[1] = state1[1]

        #down
        elif state1[0] == 5:
            # stays and connot move down
            if action1 == 3:
                state2 = state1
            # moves right
            elif action1 == 2:
                # cannot move right when it is in the [5,5]
                if state1[1] == 5:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] + 1
            # moves left
            elif action1 == 4:
                # cannot move left when it is in the [5,1]
                if state1[1] == 1:
                    state2 = state1
                else:
                    state2[0] = state1[0]
                    state2[1] = state1[1] - 1
            # moves up
            else:
                state2[0] = state1[0] - 1
                state2[1] = state1[1]

        #left 
        elif state1[1] == 1:
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
        elif state1[1] == 5:
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


#def q(state1, action1, v, gamma = 0.9):
    #state2 , r = transition(state1, action1)
    #return r + gamma * v

def returnCalc(initState, initAction, policy, gamma = 0.9, episode_length = 100, stateSpace = state_space):
    num_s_next, r = transition(initState, initAction)
    tempReturn = r
    for i in range(1, episode_length):
        num_s_next, r_next = transition(num_s_next, policy[key_of(num_s_next)])
        tempReturn += (r_next * gamma ** i)
    return tempReturn


def key_of(val):
    for k, v in state_space.items():
        if v == val:
            return k
    return None
# ---------------------------------------------------------------- pretty print
def render_box(v_pi, pi, title=None):
    """Pretty-print the 5x5 grid as aligned boxes."""
    W = 11  # cell width

    if title:
        bar = "─" * max(1, 55 - len(title))
        print(f"\n  ── {title} {bar}")

    # 5-column table borders
    top = "  ╔" + ("═" * W + "╦") * 4 + "═" * W + "╗"
    mid = "  ╠" + ("═" * W + "╬") * 4 + "═" * W + "╣"
    bot = "  ╚" + ("═" * W + "╩") * 4 + "═" * W + "╝"

    # action code → arrow (inverse of action_space)
    inv = {v: k for k, v in action_space.items()}

    # special cells (positions are 1-indexed, matching your comments)
    PITS  = {(2, 2), (2, 3), (3, 3), (4, 2), (4, 4), (5, 2)}
    GOALS = {(4, 3)}

    def cell(row, col):
        s = f"s{row * 5 + col + 1}"          # s1..s25
        v = float(v_pi[s])
        a = inv[int(pi[s])]
        # small suffix so pits / goal pop out at a glance
        if (row + 1, col + 1) in PITS:
            s = s + "·P"
        elif (row + 1, col + 1) in GOALS:
            s = s + "·G"
        return (
            f"{s:^{W}}",
            f"{('v = ' + format(v, '>7.2f')):^{W}}",
            f"{a:^{W}}",
        )

    grid = [[cell(r, c) for c in range(5)] for r in range(5)]

    print(top)
    for r, row in enumerate(grid):
        for line in range(3):                # name / value / action
            print("  ║" + "║".join(row[c][line] for c in range(5)) + "║")
        if r < 4:
            print(mid)
    print(bot)


pi_s = {key : 5 for key in state_space} # initial guess for the policy.

k_max = 10  # you can change it in [4, inf] not very much because it can harm your computer
episodeLength = 100  # you can change it in [1, x] , x >= 15
gamma = 0.9

v_s = {key: 0.0 for key in state_space} #initial guess for the state

print("═" * 60)
print("  Monte Carlo Basic  •  FrozenLake 5x5  ")
print("═" * 60)
print(f"  Discount factor γ   : {gamma}")
print(f"  Total iterations    : (pending)")

k = 0
render_box(v_s, pi_s, title=f"Initial guess  (k = {k})")
while k < (k_max):
    for s_first, num_s_first in state_space.items():
        #multipReturn = 0
        q_vals = []
        for sign, action in action_space.items():
            g = returnCalc(num_s_first, action, pi_s, gamma, episodeLength)
            # policy evaluation
            q_vals.append(g)

        # policy improvement
        a_max = np.argmax(q_vals) + 1
        pi_s[s_first] = a_max
        v_s[s_first] = np.max(q_vals)
    render_box(v_s, pi_s, title=f"After iteration:(k = {k})")
    k += 1


print()
print("═" * 60)
print(f"  Converged after {k} MC Basic steps")
print("═" * 60)
render_box(v_s, pi_s, title="Final optimal policy & state values")
print()




