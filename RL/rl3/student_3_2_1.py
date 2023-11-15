#!/usr/bin/env python3
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
rewards = [10, -10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10]

# Q learning learning rate
# not learning basiclly
alpha = 0.00001

# Q learning discount rate
gamma = 0.00001

# 下面三个是影响exploration和exploitation之间的平衡
# epsilon 本身就是exploration rate， 是一个probability[0,1], 
# 1 - exploration
# -> 0 - exploitation， 在对环境了解之后，epsilon会越来越小，代表着explore的可能性就越来越小
# Epsilon initial
epsilon_initial = 1

# do explore at the final 
# Epsilon final
epsilon_final = 1

# 是指某个参数或hyperparameter随着时间的推移逐渐减小或变化的过程。是一个启发式的优化算法。
# 这里的退火时间步是指在这段时间内，某个超参数(epsilon)会从初始值逐渐见到最小值。
# 用于 exploration 和 exploitation的平衡。
# 在epsilon greedy 中，以在开始时更多的进行探索，然后逐渐增加堆已知动作的利用
# 快速的退火会更强调探索，慢速的退货会强调利用

# 1 - the smallset annealing_timesteps，相当于没有退火, 代表hyperparameter将在较短时间内从epsilon-initial到epsilon_final
# 使得agent更快的减少探索，并增加对已知最佳action的利用，使得agent更早的稳定在一个strategy上，但可能会错过一些潜在的更好的strategy
# 如果选择更大的annealing timesteps 会使得超参数的变化更平滑，因为他们在更多的时间内逐渐减小。 可以减少训练中的不稳定性，但也可能导致agent在开始时过度探索，导致错过更好的strategy

# Annealing timesteps
annealing_timesteps = 1

# threshold
threshold = 1e-6
