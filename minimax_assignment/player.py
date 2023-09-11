#!/usr/bin/env python3
import random
import math

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


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


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        random_move = random.randrange(5)
        return ACTION_TO_STR[random_move]

    def minimax(self, node, miniPlayer, maxPlayer):
        state = node.state
        
        # depth == 0 or node is a terminal value of node
        if node.depth == 2 or len(state.fish_position) == 0:
            return self.heuristic(state)
        
        # maxPlayer
        if state.player: 
            maxEval = -math.inf
            children = node-compute_and_get_children()
            for child in children:
                maxEval = max(maxEval, self.minimax(child, miniPlayer, maxPlayer))
                if miniPlayer <= maxPlayer: #alpfa_beta_pruning
                    break
            return maxEval
        # minPlay
        else: 
            minEval = math.inf
            children = node.compute_and_get_children()
            for child in children:
                minEval = min(minEval, self.minimax(child, miniPlayer, maxPlayer))
                if miniPlayer <= maxPlayer:#alpha_beta_pruning
                    break
            return minEval

def heuristic(self, state):
    max_score = state.player_scores[0]
    min_score = state.player_scores[1]
    heuristic = max_score - min_score # 自己设置的h(x)

    #min
    if state.player:
        heuristic = heuristic + self.closet_fish(state.player, state)
    #max
    else:
        heuristic = heuristic - self.closet_fish(state.player, state)
    return heuristic

def closest_fish(self, player, state):
    hook_position = state.hook_positions[player]
    min_distance = math.inf
    for fish_position in state.fish_positions.values():
        min_distance = min(min_distance, self.get_distance(fish_position, hook_position))
    return min_distance
    
def get_distance(self, fish_position, hook_position):
    distance = math.sqrt((fish_position[0]-hook_position[0])**2 + (fish_position[1]-hook_position[1])**2)
    return distance
