
# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Initialize the q values by assigning a new dict to it
        # A Counter is a dict with default 0
        self.Q_value = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # This function returns 0 if there is no possible actions, this means we
        # haven't seen a state yet. If there is possible actions we return the
        # Q node value.

        if not self.getLegalActions(state):
            return 0.0
        else:
            return self.Q_value[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)

        #if no legal actions return 0.0
        if not legalActions:
            return 0.0

        #otherwise we assign value to a really small negative number as a default
        else:
            maxValue = float('-inf')

        #next we update value to the highest value from all states within reach
        for action in legalActions: # for each action
            Qval = self.getQValue(state, action) #get Q value of that action
            maxValue = max(maxValue, Qval) #take max of previous value and Qval
        return maxValue #finally return the max value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)

        #if no legal actions return None
        if not legalActions:
            return None

        bestAction = [] # no best action yet make blank

        #get the action with the maxValue from the rest of the Qnodes
        QValue = self.computeValueFromQValues(state)

        # for each legal action
        for action in legalActions:
            Qval = self.getQValue(state, action) #get Qvalue of this state/action
            if QValue == Qval: #if they match add to list of best actions
                bestAction.append(action)
        return random.choice(bestAction) # choose a best action at random

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        epsilon = self.epsilon #define epsilon for this function
        legalActions = self.getLegalActions(state) #list of possible actions

        #if no legal actions return None
        if not legalActions:
            return None
        else:
            # with a small probability we act randomly
            if util.flipCoin(epsilon): # with a probability of epsilon
                return random.choice(legalActions) # we return a random action

            # or we return the best action with probability of 1 - epsilon
            else:
                return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        """ Update's the Q-Value (Qk+1) I designed this to match slide 57
        """
        gamma = self.discount   #discount
        alpha = self.alpha      #learning rate
        max_qk_sPrime_a = self.computeValueFromQValues(nextState)
        sample = reward + (gamma * max_qk_sPrime_a) #new sample
        qk_sa = self.Q_value[(state, action)]   #old sample (new sample later)

        # Incorporate the new estimate into a running average
        self.Q_value[(state, action)] = qk_sa + alpha * (sample - qk_sa)

        # or this is eqivalent
        # self.Q_value[(state, action)] = ((1 - alpha) * qk_sa) + (alpha * sample)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Q(s,a) = w1f1(s,a) + w2f2(s,a)+ ... + wnfn(s,a)
        # this function calculater the wnfn(s,a) given the state/action pair
        # get weights and the vector of features and return product

        w = self.getWeights()  # get the weight
        f = self.featExtractor.getFeatures(state, action) # get the feature
        q_s_a = w * f # compute the product
        return q_s_a # return the product

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # This function updates the weights of active features for the qlearner!
        # I wrote this to match the linear Q-functions on slide 68
        # The comments describe what part of the algorithim the code represents
        gamma = self.discount   #discount
        alpha = self.alpha      #learning rate
        # max_qk_sPrime_aPrime == maxQ(s',a')
        max_qk_sPrime_aPrime = self.computeValueFromQValues(nextState)
        stuff = reward + (gamma * max_qk_sPrime_aPrime) # == [r + y(maxQ(s',a'))]
        q_sa = self.getQValue(state, action)   # == Q(s,a)
        difference = stuff - q_sa # == [r + y(maxQ(s',a'))] - Q(s,a)
        features = self.featExtractor.getFeatures(state, action) #get the features

        # for each feature update the weights
        for f in features:
            self.weights[f] += alpha * difference * features[f]
        return

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
