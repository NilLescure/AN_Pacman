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
    An offensive reflex agent that takes one food and immediately returns home.
    """

    def choose_action(self, game_state):
        """
        Overrides the default action selection to enforce one-food-and-return logic.
        """
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carried_food = my_state.num_carrying

        if carried_food > 0:
            # If carrying food, only consider actions that reduce distance to home
            best_action = None
            min_home_distance = float('inf')

            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_pos = successor.get_agent_state(self.index).get_position()
                home_distance = self.get_maze_distance(successor_pos, self.start)

                if home_distance < min_home_distance:
                    min_home_distance = home_distance
                    best_action = action

            return best_action  # Enforce return to home

        else:
            # If not carrying food, prioritize the nearest food
            food_list = self.get_food(game_state).as_list()
            if len(food_list) > 0:
                min_food_distance = float('inf')
                best_action = None

                for action in actions:
                    successor = self.get_successor(game_state, action)
                    successor_pos = successor.get_agent_state(self.index).get_position()
                    distance_to_food = min(
                        [self.get_maze_distance(successor_pos, food) for food in food_list]
                    )

                    if distance_to_food < min_food_distance:
                        min_food_distance = distance_to_food
                        best_action = action

                return best_action  # Move toward nearest food

        # If no other action, choose a random legal action as a fallback
        return random.choice(actions)

    def get_features(self, game_state, action):
        """
        Compute features for evaluation.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        carried_food = successor.get_agent_state(self.index).num_carrying

        if carried_food > 0:
            # Distance to home becomes the only feature
            home_dist = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = home_dist
        else:
            # Distance to nearest food becomes the only feature
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0:
                min_food_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_food_distance

        return features

    def get_weights(self, game_state, action):
        """
        Adjust weights to enforce strict behavior.
        """
        carried_food = game_state.get_agent_state(self.index).num_carrying

        if carried_food > 0:
            # When carrying food, only consider returning home
            return {'distance_to_home': -1}  # Minimize distance to home
        else:
            # When not carrying food, only consider getting food
            return {'distance_to_food': -1}  # Minimize distance to food


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A defensive agent that stays near the center of the defensive area and actively intercepts invaders.
    """

    def register_initial_state(self, game_state):
        """
        Initialize the agent's central defensive position.
        """
        super().register_initial_state(game_state)

        # Define a central defensive position
        self.mid_defensive_position = self.get_defensive_center(game_state)

    def get_defensive_center(self, game_state):
        """
        Compute a central position for defense based on the team's side.
        """
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        mid_x = layout_width // 2

        if self.red:
            mid_x -= 1  # Red team defends the left side
        else:
            mid_x += 1  # Blue team defends the right side

        # Find a central y-coordinate not blocked by walls
        central_y = layout_height // 2
        while game_state.has_wall(mid_x, central_y):
            central_y += 1

        return (mid_x, central_y)

    def choose_action(self, game_state):
        """
        Choose the best action to maintain a defensive position or intercept invaders.
        """
        actions = game_state.get_legal_actions(self.index)

        # Evaluate actions based on features and weights
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Choose randomly among the best actions
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        """
        Compute features for defensive behavior.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_position(self.index)

        # Defensive position feature: distance to the central defensive position
        features['distance_to_center'] = self.get_maze_distance(my_pos, self.mid_defensive_position)

        # Check for visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Distance to the closest invader
            invader_distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(invader_distances)
        else:
            # Penalize straying far from the center when no invaders are present
            features['distance_to_center'] = self.get_maze_distance(my_pos, self.mid_defensive_position)

        # Discourage stopping or reversing direction unnecessarily
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Adjust weights for defensive behavior.
        """
        weights = {
            'num_invaders': -1000,      # Strongly prioritize intercepting invaders
            'invader_distance': -10,   # Minimize distance to invaders
            'distance_to_center': -5,  # Stay close to the defensive center
            'stop': -100,              # Penalize stopping
            'reverse': -2,             # Penalize reversing direction
        }

        # Dynamically increase priority for invaders if any are detected
        if len([a for a in self.get_opponents(game_state) if game_state.get_agent_state(a).is_pacman]):
            weights['distance_to_center'] = 0  # Ignore center if invaders are present

        return weights
