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
        We override the default action selection to apply our logic
        """

        # We initialize varaibles
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carried_food = my_state.num_carrying

        # If the agent is carrying food, we only consider actions that reduce distance to home
        if carried_food > 0:
            
            # We initialize varaibles
            best_action = None
            min_home_distance = float('inf')

            # We iterate through all the actions and return the action that minimizes the distance to home
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_pos = successor.get_agent_state(self.index).get_position()
                home_distance = self.get_maze_distance(successor_pos, self.start)

                if home_distance < min_home_distance:
                    min_home_distance = home_distance
                    best_action = action

            return best_action

        # If the agent is NOT carrying food, we prioritize the nearest food
        else:

            # We get the food list
            food_list = self.get_food(game_state).as_list()

            # If there is some food to eat we return the best action to minimize the distance to the food
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

                return best_action

        # If we have not return an action we choose a random legal action as a fallback
        return random.choice(actions)

    def get_features(self, game_state, action):
        """
        We compute features for evaluation
        """

        # We initialize varaibles
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        carried_food = successor.get_agent_state(self.index).num_carrying

        # If the agent is carring food, the distance to home becomes the only feature
        if carried_food > 0:
            home_dist = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = home_dist
        
        # If the agent is NOT carring food, the distance to nearest food becomes the only feature
        else:
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0:
                min_food_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_food_distance

        return features

    def get_weights(self, game_state, action):
        """
        We adjust weights to enforce strict behavior
        """

        # We initialize the carried food
        carried_food = game_state.get_agent_state(self.index).num_carrying

        # If the agent is carring food, we only consider returning home
        if carried_food > 0:
            # We minimize distance to home
            return {'distance_to_home': -1}
        
        # If the agent is NOT carring food, we only consider getting food
        else:
            # We minimize distance to food
            return {'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    We define a defensive agent that stays near the center of the defensive area and actively intercepts invaders
    """

    def register_initial_state(self, game_state):
        """
        We initialize the agent's central defensive position
        """
        super().register_initial_state(game_state)

        # We define a central defensive position
        self.mid_defensive_position = self.get_defensive_center(game_state)

    def get_defensive_center(self, game_state):
        """
        We compute a central position for defense based on the team's side.
        """

        # We initialize varaibles
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        mid_x = layout_width // 2

        # Depending on the color of our team we defend left or right side

        # If we are red team we defend the left side
        if self.red:
            mid_x -= 1
        # If we are blue team we defend the right side
        else:
            mid_x += 1

        # We find a central y-coordinate not blocked by walls
        central_y = layout_height // 2
        while game_state.has_wall(mid_x, central_y):
            central_y += 1

        # We return the coordinate x and y where we want to defend
        return (mid_x, central_y)

    def choose_action(self, game_state):
        """
        We choose the best action to maintain a defensive position or intercept invaders
        """

        # We initialize actions
        actions = game_state.get_legal_actions(self.index)

        # We evaluate actions based on features and weights
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # We choose randomly among the best actions
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        """
        We compute features for defensive behavior.
        """

        # We initialize varaibles
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_position(self.index)

        # We define the defensive position feature (distance to the central defensive position)
        features['distance_to_center'] = self.get_maze_distance(my_pos, self.mid_defensive_position)

        # We check for visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        features['num_invaders'] = len(invaders)

        # If there is one (or more) invaders, we get the distance to the invader
        if len(invaders) > 0:
            invader_distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(invader_distances)
        # If there is NOT invaders, we penalize straying far from the center
        else:
            features['distance_to_center'] = self.get_maze_distance(my_pos, self.mid_defensive_position)

        # We discourage stopping or reversing direction unnecessarily
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # We return the features
        return features

    def get_weights(self, game_state, action):
        """
        We adjust weights for defensive behavior.
        """

        # We define the weights
        weights = {
            'num_invaders': -1000,     # We strongly prioritize intercepting invaders
            'invader_distance': -10,   # We minimize distance to invaders
            'distance_to_center': -5,  # We stay close to the defensive center
            'stop': -100,              # We penalize stopping
            'reverse': -2,             # We penalize reversing direction
        }

        # Dynamically, we increase priority for invaders if any are detected
        if len([a for a in self.get_opponents(game_state) if game_state.get_agent_state(a).is_pacman]):
            # We ignore center if invaders are present
            weights['distance_to_center'] = 0

        # We return the weights
        return weights
