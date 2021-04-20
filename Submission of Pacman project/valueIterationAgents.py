# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
            Note: these are described in mdp.py, defined in gridworld.py
            Note: these pull needed values from the mdp object
              mdp.getStates()
                        Return a list of all states in the MDP.
              mdp.getPossibleActions(state)
                        Returns list of actions from state
              mdp.getTransitionStatesAndProbs(state, action) t(s,a,s')
                        Returns list of (nextState, prob) pairs
                            Note: if terminal state returns []
              mdp.getReward(state, action, nextState)
                        Returns reward from t(s,a,s')
              mdp.isTerminal(state)
                        Returns true if the current state is a terminal state.
        """
        #compared to algorithim in textbook Fig 17.4
        self.mdp = mdp             # input mdp
        self.discount = discount   # input gamma
        self.iterations = iterations # number of iterations defined by user
        self.values = util.Counter() # --U-- A Counter is a dict with default 0
        self.runValueIteration()   # function VALUE-ITERATION call to run

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        """ This value iteration function checks that the state is not a
        terminal state and then runs a loop for each iteration of the algorithim
        calculating the maximum value of the possible actions and filling a vector
        with the q values. It calls getAction and getQValue which I define later.
        These two functions do all of the calculations necessary to determin them
        best action to take for a optimum policy. """

        # runs this loop (-i ...) specified in input
        for i in range(self.iterations): # for each iteration

            # get a new dict for this iteration U <- U'
            newStateValues = self.values.copy()

            s = self.mdp.getStates()
            # for each state in s do
            for state in s:

                terminal = self.mdp.isTerminal(state)
                # if the current state is NOT terminal then compute action/Qval
                if not terminal:
                    action = self.getAction(state) # get max action
                    newStateValues[state] = self.getQValue(state, action)

            # re populate the original dict with the new values
            self.values = newStateValues

            # completed value iteration
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        """
            This function uses the Bellman equation to calculate the utility
            of a q-state. First we get the state (called nextState) that is
            reachable by taking an action from the given state along with
            it's probability of occuring. Next we find the reward of going to
            that next state. Next, we calculate the utility using the Bellman
            equation to find the utility (currentStateValue) of k+1 for the
            input state.
        """

        currentStateValue = 0  # value starts at zero

        for nextState, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, nextState) # get reward
            y = self.discount   # get discount (gamma)
            nextU = self.getValue(nextState) # get the max utility of next state
            currentStateValue += p * (r + y * nextU) # calculate U of k+1 given s
        return currentStateValue
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        """
            This function checks if the state is terminal and if it is
            then returns None as in the instructions. Next, it calculates the
            values of each action and stores them in a temporary dict for
            comparison. Next, the function argMax() defined in util.py is used
            to find the maximum value to be returned which is eventually
            multiplied with gamma in the Bellman equation.
        """
        # if there aren't any legal actions, return None (break ties)
        if self.mdp.isTerminal(state):
          return None

        # create a temp dict for comparing actions
        tempCache = util.Counter()

        # get the optimum policy value of each state/action/Qval
        for action in self.mdp.getPossibleActions(state):
            tempCache[action] = self.getQValue(state, action)
        # use argmax to find best policy
        # (returns key with highest value defined in util.py line 334)
        policy = tempCache.argMax()
        return policy # return the best action from the given state

        # util.raiseNotDefined()
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

        #these two functions just call the functions I defined.
    def getAction(self, state):
        # "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
