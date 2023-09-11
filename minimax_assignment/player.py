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

        children = initial_tree_node.compute_and_get_children()
        best_score = -math.inf
        best_move = 0
        for child in children:
            score = self.alpha_beta_pruning(child, child.depth, math.inf, -math.inf, 0)
            if score > best_score :
                best_score = score
                best_move = child.move
        return ACTION_TO_STR[best_move]
        

    def alpha_beta_pruning(self, node, depth, beta, alpha, maxPlayer):
        state = node.state
        children = node.compute_and_get_children()
        # depth = node.depth
        # depth == 0 or node is a terminal value of node
        if depth == 0 or len(children) == 0: 
            return self.heuristic(state)
        
        # alpha
        # maxPlayer = state.player
        if maxPlayer == 0:  
            value = -math.inf
            for child in children:
                value = max(value, self.alpha_beta_pruning(child, depth - 1, beta, alpha, 1)) 
                if value > beta: 
                    break
                alpha = max(alpha, value)
            return value
        # beta
        else: 
            value = math.inf
            children = node.compute_and_get_children()
            for child in children:
                value = min(value, self.alpha_beta_pruning(child, depth - 1, beta, alpha, 0))
                if value < alpha:
                    break
                beta = min(beta, value)
            return value

    def heuristic(self, state):
        heuristic = state.player_scores[0] - state.player_scores[1]

        hook_position = state.hook_positions[state.player]
        min_distance = math.inf
        for fish_position in state.fish_positions.values():
            distance = math.sqrt((fish_position[0] - hook_position[0])**2 + (fish_position[1] - hook_position[1])**2)
            min_distance = min(min_distance, distance)
        if state.player:
            heuristic = heuristic + min_distance
        else:
            heuristic = heuristic - min_distance
        return heuristic