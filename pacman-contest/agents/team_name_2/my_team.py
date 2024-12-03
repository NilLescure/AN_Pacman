# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers. isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Our offensive reflex agent will take one food and immediately returns home, then he return to search food
    """
    def choose_action(self, game_state):
        """
        Chooses an action based on the current game state.
        """
        # We initialize some variables
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food = self.get_food(game_state).as_list()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        ghosts = [agent for agent in enemies if not agent.is_pacman and agent.get_position() is not None]
        scared_ghosts = [ghost for ghost in ghosts if ghost.scared_timer > 0]
        attaking_ghosts = [ghost for ghost in ghosts if ghost not in scared_ghosts]

        food_carring = my_state.num_carrying

        # We see if there is any food near
        food_near = any(self.get_maze_distance(my_pos, food_pos) <5 for food_pos in food)

        # If the agent is not carring food or there is food near we do the following
        if food_carring == 0 or food_near:
            
            # We calculate the closest food
            food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) for food_pos in food]
            closest_food = min(food_distances, key=lambda x: x[0])[1]
        
            # We compute the distance to the ghosts
            ghosts_distances = [(self.get_maze_distance(my_pos, ghost.get_position()), ghost.get_position())for ghost in attaking_ghosts if ghost.get_position()]
            
            if ghosts_distances:
                # If there is ghosts we compute the min distance to a ghost
                closest_ghost_dist = min(ghosts_distances, key=lambda x: x[0])[0]
                
                if closest_ghost_dist < 3 and food_carring != 0:
                    # If the min distance to a ghost is less than 3 and the agent is carring food we return the best actions to go to our side
                    best_action = None
                    min_distance = float('inf')
                    for action in actions:
                        successor = self.get_successor(game_state, action)
                        new_pos = successor.get_agent_state(self.index).get_position()
                        distance = self.get_maze_distance(new_pos, self.start)
                        if distance < min_distance:
                            min_distance = distance
                            best_action = action
                    return best_action
                        
            # We return the best action to go to the closest food being carefull with the ghosts
            best_action = None
            min_distance = float('inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                distance = self.get_maze_distance(new_pos, closest_food)
                ghosts_distances = [self.get_maze_distance(new_pos, ghost.get_position()) for ghost in attaking_ghosts if ghost.get_position()]
                if distance < min_distance and all(dist > 1 for dist, ghost in zip(ghosts_distances, attaking_ghosts)):
                    min_distance = distance
                    best_action = action
            if best_action:
                return best_action

        # If the agent is carring food and he does not have food near, we return the best action to go our side
        best_action = None
        min_distance = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(new_pos, self.start)
            if distance < min_distance:
                min_distance = distance
                best_action = action
        return best_action


class DefensiveReflexAgent(ReflexCaptureAgent):

    def get_defensive_center(self, game_state):
        """
        We compute a central position for defense based on the team's side.
        """

        # We initialize varaibles
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        mid_x = layout_width // 2

        # We find a central y-coordinate not blocked by walls
        central_y = layout_height // 2
        while game_state.has_wall(mid_x, central_y):
            central_y += 1
        
        # Depending on the color of our team we defend left or right side

        # If we are red team we defend the left side
        if self.red:
            mid_x -= 2
        # If we are blue team we defend the right side
        else:
            mid_x += 1
            central_y -= 1

        # We return the coordinate x and y where we want to defend
        return (mid_x, central_y)

    def get_features(self, game_state, action):
        """
        We compute the features.
        """
        # We initialize some features
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # We define the defense status, if it is a ghost or a pacman
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # We compute the number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # FWe compute the distance to the nearest invader
        if invaders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            features['invader_distance'] = 0

        # We compute the distance to nearest food being defended
        food_list = self.get_food_you_are_defending(successor).as_list()
        if food_list:
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        # We compute the distance to the central defensive position
        features['distance_to_center'] = self.get_maze_distance(my_pos, self.get_defensive_center(game_state))

        # We define the stop and reverse
        features['stop'] = 1 if action == Directions.STOP else 0
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        successor_direction = successor.get_agent_state(self.index).configuration.direction
        features['reverse'] = 1 if successor_direction == Directions.REVERSE[current_direction] else 0

        # We return the features
        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,     # We give high priority to block invaders
            'on_defense': 1000,        # We give high priority to staying on defense
            'invader_distance': -20,   # We give strong penalty for being far from invaders
            'distance_to_food': -1,    # We give a minor penalty for distance to food
            'distance_to_center': -2,  # We encourage staying near the center
            'stop': -100,              # We give penalty for stopping
            'reverse': -5              # We give slight penalty for reversing
        }
