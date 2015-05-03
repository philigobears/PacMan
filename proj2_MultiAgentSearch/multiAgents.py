# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from graphicsDisplay import FOOD_COLOR

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        "*** YOUR CODE HERE ***"
        # CURRENT INFO
        pos = currentGameState.getPacmanPosition()
        currentGhostStates = currentGameState.getGhostStates()
        currentGhostPos = [gho.getPosition() for gho in currentGhostStates]
        currentGhostDist = [util.manhattanDistance(pos, p) for p in currentGhostPos]
        currentMinGhostDist = min(currentGhostDist)
        currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
        currentCapsuleList = currentGameState.getCapsules()
        # NEW INFO
        newGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = newGameState.getPacmanPosition()
        newGhostStates = newGameState.getGhostStates()
        newGhostPos = [gho.getPosition() for gho in newGhostStates]
        newGhostDist = [util.manhattanDistance(newPos, p) for p in newGhostPos]
        newMinGhostDist = min(newGhostDist)
        newFood = newGameState.getFood()
        newFoodList = newFood.asList()
        newFoodDist = [util.manhattanDistance(newPos, pos) for pos in newFoodList]
        newMinFoodDist = min(newFoodDist) if newFoodList else 0
        newScore = newGameState.getScore()
        # CASE 1, ATTACK GHOSTS
        if max(currentScaredTimes) > 0 and max(currentScaredTimes) >= currentMinGhostDist: ### TO BE MODIFIED
            # print "attack"
            return 999999 - newMinGhostDist
        # CASE 2, AVOID GHOSTS
        if newMinGhostDist < 2:
            # print "avoid"
            return -999999
        # CASE 3, MOVE FORWARD
        if action == 'Stop':
            # print "stop"
            return -999998
        # CASE 4, REACH CALSULE
        if currentCapsuleList:
            newCapsuleDist = [util.manhattanDistance(newPos, p) for p in currentCapsuleList]
            newMinCapsuleDist = min(newCapsuleDist)
            # print "arm"
            return 999999 - newMinCapsuleDist
        # CASE 5, REACH FOOD
        # print "food"
        return newScore + float(1) / (newMinFoodDist+1)
            


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.getUtility(gameState, 0, self.depth)[1]
    
    def getUtility(self, gameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0:
            return (self.evaluationFunction(gameState), None)
        if agentIndex == 0:
            return self.maxHelper(gameState, agentIndex, depth)
        else:
            return self.minHelper(gameState, agentIndex, depth)
        
    def maxHelper(self, gameState, agentIndex, depth):
        utility, action = -999999, None
        if gameState.isLose() or gameState.isWin():
            utility = self.evaluationFunction(gameState)
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for newAction in legalActions:
                newGameState = gameState.generateSuccessor(agentIndex, newAction)
                newUtility = self.getUtility(newGameState, agentIndex + 1, depth)[0]
                if newUtility > utility:
                    utility, action = newUtility, newAction
        return (utility, action)

    def minHelper(self, gameState, agentIndex, depth):
        utility, action = 999999, None
        if gameState.isLose() or gameState.isWin():
            utility = self.evaluationFunction(gameState)
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for newAction in legalActions:
                newGameState = gameState.generateSuccessor(agentIndex, newAction)
                newUtility = self.getUtility(newGameState, agentIndex + 1, depth)[0]
                if newUtility < utility:
                    utility, action = newUtility, newAction
        return (utility, action)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getUtility(gameState, 0, self.depth, -999999, 999999)[1]
    
    def getUtility(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0:
            return (self.evaluationFunction(gameState), None)
        if agentIndex == 0:
            return self.maxHelper(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minHelper(gameState, agentIndex, depth, alpha, beta)
        
    def maxHelper(self, gameState, agentIndex, depth, alpha, beta):
        utility, action = -999999, None
        if gameState.isLose() or gameState.isWin():
            utility = self.evaluationFunction(gameState)
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for newAction in legalActions:
                newGameState = gameState.generateSuccessor(agentIndex, newAction)
                newUtility = self.getUtility(newGameState, agentIndex + 1, depth, alpha, beta)[0]
                if newUtility > utility:
                    utility, action = newUtility, newAction
                if utility > beta:
                    return (utility, action)
                alpha = max(alpha, utility)
        return (utility, action)

    def minHelper(self, gameState, agentIndex, depth, alpha, beta):
        utility, action = 999999, None
        if gameState.isLose() or gameState.isWin():
            utility = self.evaluationFunction(gameState)
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for newAction in legalActions:
                newGameState = gameState.generateSuccessor(agentIndex, newAction)
                newUtility = self.getUtility(newGameState, agentIndex + 1, depth, alpha, beta)[0]
                if newUtility < utility:
                    utility, action = newUtility, newAction
                if utility < alpha:
                    return (utility, action)
                beta = min(beta, utility)
        return (utility, action)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getUtility(gameState, 0, self.depth)[1]
    
    def getUtility(self, gameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0:
            return (self.evaluationFunction(gameState), None)
        if agentIndex == 0:
            return self.maxHelper(gameState, agentIndex, depth)
        else:
            return self.expHelper(gameState, agentIndex, depth)
        
    def maxHelper(self, gameState, agentIndex, depth):
        utility, action = -999999, None
        if gameState.isLose() or gameState.isWin():
            utility = self.evaluationFunction(gameState)
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for newAction in legalActions:
                newGameState = gameState.generateSuccessor(agentIndex, newAction)
                newUtility = self.getUtility(newGameState, agentIndex + 1, depth)[0]
                if newUtility > utility:
                    utility, action = newUtility, newAction
        return (utility, action)

    def expHelper(self, gameState, agentIndex, depth):
        utility, action = 0, None
        if gameState.isLose() or gameState.isWin():
            utility = self.evaluationFunction(gameState)
        else:
            legalActions = gameState.getLegalActions(agentIndex)
            for newAction in legalActions:
                newGameState = gameState.generateSuccessor(agentIndex, newAction)
                utility += self.getUtility(newGameState, agentIndex + 1, depth)[0]
            utility = float(utility) / len(legalActions)
        return (utility, None)
    


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    wall = currentGameState.getWalls()
    score = currentGameState.getScore()
    
    ghostStates = currentGameState.getGhostStates()
    ghostPos = [gho.getPosition() for gho in ghostStates]
    ghostDist = [util.manhattanDistance(pos, p) for p in ghostPos]
    minGhostDist = min(ghostDist)
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    minscaredTimes = min(scaredTimes)
    
    capsuleList = currentGameState.getCapsules()
    capsuleDist = [util.manhattanDistance(pos, p) for p in capsuleList]
    minCapsuleDist = min(capsuleDist) if capsuleDist else 0
    minCapsule = None
    for capsule in capsuleList:
        if util.manhattanDistance(pos, capsule) == minCapsuleDist:
            minCapsule = capsule
            break
    
    foodList = currentGameState.getFood().asList()
    foodDist = [util.manhattanDistance(pos, p) for p in foodList]
    minFoodDist = min(foodDist) if foodDist else 0
    minFood = None
    for food in foodList:
        if util.manhattanDistance(pos, food) == minFoodDist:
            minFood = food
            break
    
    # CASE 1: IF NOT SCARED AND HAS CAPSULE, GIVE PRIORITY TO CAPSULE
    if minscaredTimes == 0 and capsuleList:
        dist = aStarDisCalculator(pos, minCapsule, wall)
    # CASE 2: IF SCARED OR NO CAPSULE, GIVE PRIORITY TO FOOD
    elif minFood:
        dist = aStarDisCalculator(pos, minFood, wall)
    # CASE 3: ELSE
    else:
        dist = 0
    return score + float(1) / (dist + 1)

def aStarDisCalculator(start, end, wall):
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push((start, 0), util.manhattanDistance(start, end))
    while not fringe.isEmpty():
        pos, dist = fringe.pop()
        if util.manhattanDistance(pos, end) == 0:
            return dist
        if pos not in closed:
            closed.add(pos)
            x, y = pos
            for newPos in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if not wall[newPos[0]][newPos[1]]:
                    fringe.push((newPos, dist + 1), util.manhattanDistance(newPos, end) + dist + 1)
    return None
                
# Abbreviation
better = betterEvaluationFunction

