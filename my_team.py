# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
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
            best_dist = 9999
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
    FOOD_THRESHOLD = 2  # Collect two food pellets before returning near the defensive agent
    RETURN_REWARD = 1000  # Reward for successfully reaching near the defensive agent

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.food_carried = 0  # Track the number of food pellets collected
        self.mode = 'collect'  # Modes: 'collect' or 'return_near_defensive'

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(game_state).as_list()  # Current food list
        successor_food_list = self.get_food(successor).as_list()  # Food list after action

        # Locate the defensive agent
        defensive_agent_index = self.get_team(game_state)[1] if self.index == self.get_team(game_state)[0] else self.get_team(game_state)[0]
        defensive_agent_pos = game_state.get_agent_position(defensive_agent_index)

        # Collect mode logic
        if self.mode == 'collect':
            # Prioritize the nearest food
            if food_list:
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance

            # Check if food is eaten
            if my_pos in food_list and my_pos not in successor_food_list:
                self.food_carried += 1  # Increment food count only if food is eaten
                if self.food_carried >= self.FOOD_THRESHOLD:
                    self.mode = 'return_near_defensive'  # Transition to return mode near defensive agent

        # Return mode logic: Move towards the defensive agent
        if self.mode == 'return_near_defensive':
            # Prioritize moving toward the defensive agent
            dist_to_defensive_agent = self.get_maze_distance(my_pos, defensive_agent_pos)
            features['distance_to_defensive_agent'] = dist_to_defensive_agent

            # If near the defensive agent, reset to collect mode
            if dist_to_defensive_agent <= 2:  # Within a certain distance of the defensive agent
                features['return_bonus'] = 1
                self.mode = 'collect'  # Reset to collect mode
                self.food_carried = 0  # Reset food count

        # Ghost avoidance
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            min_ghost_distance = min(ghost_dists)

            # Penalize being near ghosts
            if min_ghost_distance <= 2:  # Danger zone
                features['ghost_proximity'] = 1  # Immediate escape
            features['ghost_distance'] = min_ghost_distance

        return features

    def get_weights(self, game_state, action):
        # Adjust weights dynamically based on mode
        if self.mode == 'collect':
            weights = {
                'distance_to_food': -10,       # Strong preference for closer food
                'ghost_distance': 2,          # Reward keeping distance from ghosts
                'ghost_proximity': -1000,     # Strongly penalize being close to ghosts
            }
        elif self.mode == 'return_near_defensive':
            weights = {
                'distance_to_defensive_agent': -10,  # Strong preference for getting closer to the defensive agent
                'return_bonus': self.RETURN_REWARD,  # Reward for reaching near defensive agent
                'ghost_distance': 2,          # Reward keeping distance from ghosts
                'ghost_proximity': -1000,     # Strongly penalize being close to ghosts
            }
        return weights



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Defensive posture
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Invader tracking
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(dists)

        # Patrolling critical zones
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        if food_defending:
            min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_defending])
            features['distance_to_food'] = min_food_dist

        # Penalize stationary or reversed movement
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'distance_to_food': -5,
            'stop': -100,
            'reverse': -2
        }

