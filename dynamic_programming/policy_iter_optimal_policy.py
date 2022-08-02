# iterative policy evaluation
# gamma = float(sys.argv[3])
import json
import sys

# Load json file (MDP)
fobj = open(sys.argv[1])
jsonobj = json.load(fobj)

# Prepare state strings
num_states = jsonobj['num_states']
states = list()
for i in range(num_states):
    states.append(f's{i}')

# Tran Prob matrix
P = jsonobj["tran_prob"]

# Reward vector
R = jsonobj["rewards"]

# e.g. Transition probability from state 0 to state 0, choosing action 'r'
print(f'p_00^r = {P[states[0]]["r"][0]}')  # 0.1

# e.g. Reward of state 0
print(f'r_0 = {R[states[0]]}')  # 3.0

# Load random policy
fpobj = open(sys.argv[2])
jsonobj = json.load(fpobj)
pi = {}
for s in states:
    pi[s] = jsonobj
print(pi)
print(f'pi.s0 = {pi["s0"]}')


# -------------------------------
# V[i] represents the value of state i (si)
V = [0.0 for _ in range(num_states)]

k = 1
gamma = float(sys.argv[3])
while True:
    new_value = [0.0 for _ in range(num_states)]
    for s in range(len(states)):
        vs = 0
        # action_value = [0 for i in range(len(V))]
        for a in P[states[s]].keys():
            qsa = 0
            # next states
            for i in range(len(P[states[s]][a])):
                # the average of the successors' actions values
                qsa += P[states[s]][a][i] * V[i]
                # the average of the state value given policy
            vs += pi[states[s]][a] * (R[states[s]] + gamma * qsa)
        new_value[s] = vs

    print("iteration", k)
    print(V)
    print(new_value)
    k += 1

    keep_looping = False
    for i in range(num_states):
        if abs(V[i] - new_value[i]) > 0.001:
            keep_looping = True
    V = new_value
    if not keep_looping:
        break

    # update pi
    for s in range(len(states)):
        # action_value = [0 for i in range(len(V))]
        max_value = None
        max_action = None
        for a in P[states[s]].keys():
            qsa = 0
            # next states
            for i in range(len(P[states[s]][a])):
                # the average of the successors' actions values
                qsa += P[states[s]][a][i] * V[i]
            if max_value is None or qsa > max_value:
                max_value = qsa
                max_action = a

        for a in P[states[s]].keys():
            if a == max_action:
                pi[states[s]][a] = 1.0
            else:
                pi[states[s]][a] = 0.0

    print("------------------------")
print("----------END-----------")

# --------------------------------
# Do not change the following. The output will be graded.
print()
print(f'V: {V}')
print()
