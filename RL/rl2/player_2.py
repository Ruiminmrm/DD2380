#!/usr/bin/env python3
import random
import numpy as np

from agent import Fish
from communicator import Communicator
from shared import SettingLoader


class FishesModelling:
    def init_fishes(self, n):
        fishes = {}
        for i in range(n):
            fishes["fish" + str(i)] = Fish()
        self.fishes = fishes


class PlayerController(SettingLoader, Communicator):
    def __init__(self):
        SettingLoader.__init__(self)
        Communicator.__init__(self)
        self.space_subdivisions = 10
        self.actions = None
        self.action_list = None
        self.states = None
        self.init_state = None
        self.ind2state = None
        self.state2ind = None
        self.alpha = 0
        self.gamma = 0
        self.episode_max = 300

    def init_states(self):
        ind2state = {}
        state2ind = {}
        count = 0
        for row in range(self.space_subdivisions):
            for col in range(self.space_subdivisions):
                ind2state[(col, row)] = count
                state2ind[count] = [col, row]
                count += 1
        self.ind2state = ind2state
        self.state2ind = state2ind

    def init_actions(self):
        self.actions = {
            "left": (-1, 0),
            "right": (1, 0),
            "down": (0, -1),
            "up": (0, 1)
        }
        self.action_list = list(self.actions.keys())

    def allowed_movements(self):
        self.allowed_moves = {}
        for s in self.ind2state.keys():
            self.allowed_moves[self.ind2state[s]] = []
            if s[0] < self.space_subdivisions - 1:
                self.allowed_moves[self.ind2state[s]] += [1]
            if s[0] > 0:
                self.allowed_moves[self.ind2state[s]] += [0]
            if s[1] < self.space_subdivisions - 1:
                self.allowed_moves[self.ind2state[s]] += [3]
            if s[1] > 0:
                self.allowed_moves[self.ind2state[s]] += [2]

    def player_loop(self):
        pass


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


def epsilon_greedy(Q,
                   state,
                   all_actions,
                   current_total_steps=0,
                   epsilon_initial=1,
                   epsilon_final=0.2,
                   anneal_timesteps=10000,
                   eps_type="constant"):

    if eps_type == 'constant':
        epsilon = epsilon_final
        # ADD YOUR CODE SNIPPET BETWEEN EX 4.1
        # Implemenmt the epsilon-greedy algorithm for a constant epsilon value
        # Use epsilon and all input arguments of epsilon_greedy you see fit
        # It is recommended you use the np.random module
        action = None
        # ADD YOUR CODE SNIPPET BETWEEN EX 4.1

    elif eps_type == 'linear':
        # ADD YOUR CODE SNIPPET BETWEENEX  4.2
        # Implemenmt the epsilon-greedy algorithm for a linear epsilon value
        # Use epsilon and all input arguments of epsilon_greedy you see fit
        # use the ScheduleLinear class
        # It is recommended you use the np.random module
        action = None
        # ADD YOUR CODE SNIPPET BETWEENEX  4.2

    else:
        raise "Epsilon greedy type unknown"

    return action


class PlayerControllerRL(PlayerController, FishesModelling):
    def __init__(self):
        super().__init__()

    def player_loop(self):
        # send message to game that you are ready
        self.init_actions()
        self.init_states()
        self.alpha = self.settings.alpha
        self.gamma = self.settings.gamma
        self.epsilon_initial = self.settings.epsilon_initial
        self.epsilon_final = self.settings.epsilon_final
        self.annealing_timesteps = self.settings.annealing_timesteps
        self.threshold = self.settings.threshold
        self.episode_max = self.settings.episode_max

        q = self.q_learning()

        # compute policy
        policy = self.get_policy(q)

        # send policy
        msg = {"policy": policy, "exploration": False}
        self.sender(msg)

        msg = self.receiver()
        print("Q-learning returning")
        return

    def q_learning(self):
        ns = len(self.state2ind.keys())
        na = len(self.actions.keys())
        discount = self.gamma
        lr = self.alpha
        # initialization
        self.allowed_movements()
        # ADD YOUR CODE SNIPPET BETWEEN EX. 2.1
        # Initialize a numpy array with ns state rows and na state columns with float values from 0.0 to 1.0.
        Q = np.random.uniform(0, 1, (ns, na))
        # 这就是Q table
        # Q table： 用于表示每个状态，动作pair的估计值，即采取特定动作的预期累计奖励。
        # 1. 在初始阶段时采取随即动作，以探索环境并了解更多关于奖励和状态转换的信息。
        # 2. 同时也是为了避免算法陷入局部最优解（local optimal solution）。如果所有Q值都预设为相同的值，那么在learning过程中就会过于集中于莫一组动作，而忽略别的更好的策略
        # 3. 用随机初始化，有助于确保算法能够探索不同的动作和策略，以找到更好的策略
        # ADD YOUR CODE SNIPPET BETWEEN EX. 2.1

        for s in range(ns):
            list_pos = self.allowed_moves[s]
            for i in range(4):
                if i not in list_pos:
                    Q[s, i] = np.nan

        Q_old = Q.copy()

        diff = np.infty
        end_episode = False

        init_pos_tuple = self.settings.init_pos_diver
        init_pos = self.ind2state[(init_pos_tuple[0], init_pos_tuple[1])]
        episode = 0

        R_total = 0
        current_total_steps = 0
        steps = 0

        # ADD YOUR CODE SNIPPET BETWEEN EX. 2.3
        # Change the while loop to incorporate a threshold limit, to stop training when the mean difference
        # in the Q table is lower than the threshold
        # 第一个条件是基于训练的最大迭代次数。只要当前epsilon小于最大的epsilon的值，就会一直循环， 确保不会无限迭代，训练
        # 第二个条件是基于Q table的变化来控制训练的终止。 diff表示Q表在两次epsilon之间的平均差异（即Q值的变化）
        # 如果 两次迭代之间的变化 diff 小于或等于threhold，代表着Q值不在变化（converge）， 终止循环，即不用进行更多的训练迭代
        '''
        此时的目的是为了控制在何时停下,为了防止无限制的训练,和在Q converge时 stop the loop
        '''
        while episode <= self.episode_max and diff > self.threshold:
            # ADD YOUR CODE SNIPPET BETWEENEX. 2.3

            s_current = init_pos
            R_total = 0
            steps = 0
            while not end_episode:
                # selection of action
                list_pos = self.allowed_moves[s_current]

                # ADD YOUR CODE SNIPPET BETWEEN EX 2.1 and 2.2
                # Chose an action from all possible actions
                # 为current state选一个action，这个action是有最高Q值的action
                # 其实就是基于Q table中的信息选一个最佳的action，以便在当前状态下行动。
                # 选择具有最高Q值的动作意味着有希望获得最大的reward， 从而促进学习strategy，是Q Learning中的exploitation部分
                action = np.nanargmax(Q[s_current])
                # ADD YOUR CODE SNIPPET BETWEEN EX 2.1 and 2.2

                # ADD YOUR CODE SNIPPET BETWEEN EX 5
                # Use the epsilon greedy algorithm to retrieve an action
                # ADD YOUR CODE SNIPPET BETWEEN EX 5

                # compute reward
                action_str = self.action_list[action]
                msg = {"action": action_str, "exploration": True}
                self.sender(msg)

                # wait response from game
                msg = self.receiver()
                R = msg["reward"]
                R_total += R
                s_next_tuple = msg["state"]
                end_episode = msg["end_episode"]
                s_next = self.ind2state[s_next_tuple]

                # ADD YOUR CODE SNIPPET BETWEEN EX. 2.2
                # Implement the Bellman Update equation to update Q
                # equation :  Q(s,a) = Q(s,a) + alpha * (reward(s,a) + discount * max(Q(s',a') - Q(s,a))
                # alpha - learning rate - purpose : Q value will not update too fast, thus maintaing stability
                # reward - immediate reward of s taking a 
                # discount - use to get the converge value in the future
                # max(Q(s', a')) the action which has the highest Q value of the actions that next state gonna take 
                Q[s_current][action] = Q[s_current][action] + lr*(R + discount*np.nanmax(Q[s_next]) - Q[s_current][action])
                # ADD YOUR CODE SNIPPET BETWEEN EX. 2.2

                s_current = s_next
                current_total_steps += 1
                steps += 1

            # ADD YOUR CODE SNIPPET BETWEEN EX. 2.3
            # Compute the absolute value of the mean between the Q and Q-old
            '''
            np.mean() 和 np.nanmean()的差别
            1. mean 算 mean/average of values in the input array. 如果array 里有NaN, 会被当作regular number算进去, 结果会变NaN
            2. nanmean 会跳过NaN
            '''
            # diff - 就是上次的Q table与这次Q table的difference
            # 当diff到达一个足够小时，代表Q converge， 跟上面的while 用的条件里的diff是同一个，只是这里是update
            diff = np.nanmean(np.abs(Q - Q_old))
            # ADD YOUR CODE SNIPPET BETWEEN EX. 2.3
            Q_old[:] = Q
            print(
                "Episode: {}, Steps {}, Diff: {:6e}, Total Reward: {}, Total Steps {}"
                .format(episode, steps, diff, R_total, current_total_steps))
            episode += 1
            end_episode = False

        return Q

    def get_policy(self, Q):
        max_actions = np.nanargmax(Q, axis=1)
        policy = {}
        list_actions = list(self.actions.keys())
        for n in self.state2ind.keys():
            state_tuple = self.state2ind[n]
            policy[(state_tuple[0],
                    state_tuple[1])] = list_actions[max_actions[n]]
        return policy


class PlayerControllerRandom(PlayerController):
    def __init__(self):
        super().__init__()

    def player_loop(self):

        self.init_actions()
        self.init_states()
        self.allowed_movements()
        self.episode_max = self.settings.episode_max

        n = self.random_agent()

        # compute policy
        policy = self.get_policy(n)

        # send policy
        msg = {"policy": policy, "exploration": False}
        self.sender(msg)

        msg = self.receiver()
        print("Random Agent returning")
        return

    def random_agent(self):
        ns = len(self.state2ind.keys())
        na = len(self.actions.keys())
        init_pos_tuple = self.settings.init_pos_diver
        init_pos = self.ind2state[(init_pos_tuple[0], init_pos_tuple[1])]
        episode = 0
        R_total = 0
        steps = 0
        current_total_steps = 0
        end_episode = False
        # ADD YOUR CODE SNIPPET BETWEEN EX. 1.2
        # Initialize a numpy array with ns state rows and na state columns with zeros
        # ADD YOUR CODE SNIPPET BETWEEN EX. 1.2

        while episode <= self.episode_max:
            s_current = init_pos
            R_total = 0
            steps = 0
            while not end_episode:
                # all possible actions
                possible_actions = self.allowed_moves[s_current]

                # ADD YOUR CODE SNIPPET BETWEEN EX. 1.2
                # Chose an action from all possible actions and add to the counter of actions per state
                action = None
                # ADD YOUR CODE SNIPPET BETWEEN EX. 1.2

                action_str = self.action_list[action]
                msg = {"action": action_str, "exploration": True}
                self.sender(msg)

                # wait response from game
                msg = self.receiver()
                R = msg["reward"]
                s_next_tuple = msg["state"]
                end_episode = msg["end_episode"]
                s_next = self.ind2state[s_next_tuple]
                s_current = s_next
                R_total += R
                current_total_steps += 1
                steps += 1

            print("Episode: {}, Steps {}, Total Reward: {}, Total Steps {}".
                  format(episode, steps, R_total, current_total_steps))
            episode += 1
            end_episode = False

        return n

    def get_policy(self, Q):
        nan_max_actions_proxy = [None for _ in range(len(Q))]
        for _ in range(len(Q)):
            try:
                nan_max_actions_proxy[_] = np.nanargmax(Q[_])
            except:
                nan_max_actions_proxy[_] = np.random.choice([0, 1, 2, 3])

        nan_max_actions_proxy = np.array(nan_max_actions_proxy)

        assert nan_max_actions_proxy.all() == nan_max_actions_proxy.all()

        policy = {}
        list_actions = list(self.actions.keys())
        for n in self.state2ind.keys():
            state_tuple = self.state2ind[n]
            policy[(state_tuple[0],
                    state_tuple[1])] = list_actions[nan_max_actions_proxy[n]]
        return policy


class ScheduleLinear(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        # ADD YOUR CODE SNIPPET BETWEEN EX 4.2
        # Return the annealed linear value
        return self.initial_p
        # ADD YOUR CODE SNIPPET BETWEEN EX 4.2
