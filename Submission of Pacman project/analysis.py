# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

"""     Question Two

It seems that any value less than .016 for the noise paramerter causes the
probability of taking the action east starting from state (1,1) and endings
in state (6,1) to become positive. The original value (0.02) for noise causes
a negative probability in state (1,1) of going east. As I tested noise values,
I decreased the noise parameter by 0.005 until I started to see changes.
Slowly I watched the far right states turn green, but they all had to become
green for a reward of ten.
"""
def question2():
    answerDiscount = 0.9
    answerNoise = 0.016 # 0.2
    return answerDiscount, answerNoise

""" Question 3
I found these mainly by trial and error. You can watch the affects of each try
on the different q values for taking an action from a state and how it changes.
"""
def question3a():
    # You can use a very small value of gamma to cause the bellman equation to
    # add the value of the next states reward. Since we get to the rewards
    # faster by going right, this method works for our wanted policy type.
    answerDiscount = .1 # or even .000000001
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = .2
    answerNoise = .2
    answerLivingReward = .5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = .9
    answerNoise = .02
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = .9
    answerNoise = .5
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = .9
    answerNoise = .02
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8(): #after testing values, I determined that it is not possible.
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'

    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
