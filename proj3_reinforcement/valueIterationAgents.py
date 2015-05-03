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
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        self.cur_value = util.Counter()
        for i in range(iterations):
        	for cur_state in all_states:
        		if mdp.isTerminal(cur_state):
        			self.cur_value[cur_state] = 0
        		else:
        			actions = self.mdp.getPossibleActions(cur_state)
        			best_q = -2147483648
        			if (len(actions) == 0):
        				self.cur_value[cur_state] = 0
        			else:
        				for act in actions:
        					temp_q = 0
        					state_and_prob = self.mdp.getTransitionStatesAndProbs(cur_state, act)
        					if(len(state_and_prob)!=0):
        						for single_state in state_and_prob:
        							temp_q += single_state[1]*(self.mdp.getReward(cur_state, act, single_state[0]) + \
                                                self.discount*self.getValue(single_state[0])) 
      							if temp_q > best_q:
      								best_q = temp_q;
        				self.cur_value[cur_state] = best_q
        	self.values = self.cur_value.copy()
        
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
        state_and_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for single_state in state_and_prob:
          q_value += single_state[1]*(self.mdp.getReward(state, action, single_state[0]) + \
                        self.discount*self.getValue(single_state[0])) 
        return q_value
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if (len(actions) == 0):
          return None;
        else:
          best_act = actions[0]
          best_q = -2147483648
          for act in actions:
            temp_q = 0
            state_and_prob = self.mdp.getTransitionStatesAndProbs(state, act)
            if(len(state_and_prob)!=0):
              for single_state in state_and_prob:
                temp_q += single_state[1]*(self.mdp.getReward(state, act, single_state[0]) + \
                            self.discount*self.getValue(single_state[0])) 
              if temp_q > best_q:
                best_q = temp_q;
                best_act = act;
          return best_act
   

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
