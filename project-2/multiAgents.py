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


from typing import Dict, List, Tuple
from pacman import GameState
from util import manhattanDistance
from game import AgentState, Directions, Actions
import random, util

from game import Agent, Grid

DIRECTIONS = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
]

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState) -> str:
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        walls = successorGameState.getWalls()
        width = walls.width
        height = walls.height
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghosts = Grid(width, height)
        for i in range(len(newGhostStates)):
            if newScaredTimes[i] <= 0:
                x, y = newGhostStates[i].getPosition()
                ghosts[int(x)][int(y)] = True

        queue = util.Queue()
        queue.push((newPos, 0))
        visited = set()
        shortest = float('inf')
        ghosts_dis = []
        while not queue.isEmpty():
            cur, dis = queue.pop()
            x, y = cur
            if in_range(cur, width, height) and not walls[x][y] and cur not in visited:
                visited.add(cur)
                if newFood[x][y]:
                    shortest = min(dis, shortest)
                if ghosts[x][y]:
                    ghosts_dis.append(dis)
                for d in DIRECTIONS:
                    queue.push(((x + d[0], y + d[1]), dis + 1))
        if shortest == float('inf'):
            shortest = 0
            
        score = successorGameState.getScore()
        def d(x):
            if x == 0:
                return float('inf')
            return 9 / (x**2)
        score -= shortest + sum(map(d, ghosts_dis))
        if action == 'Stop':
            score -= 10
        return score

def in_range(pos, width, height):
    x, y = pos
    return 0 <= x and x < width and 0 <= y and y < height

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

    def getAction(self, gameState: GameState) -> str:
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
        return self._minimax(gameState, 0)[1]

    def _minimax(self, gameState: GameState, idx: int) -> Tuple[float, str]:
        n = gameState.getNumAgents()

        if idx / n >= self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

        agent = idx % n
        legalActions = gameState.getLegalActions(agent)

        mod = 1 if agent == 0 else -1
        best_score = -float('inf') * mod
        best_action = None
        for legalAction in legalActions:
            s = gameState.generateSuccessor(agent, legalAction)
            score = self._minimax(s, idx + 1)[0]
            if score * mod > best_score * mod:
                best_score, best_action = score, legalAction
        
        return (best_score, best_action)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState) -> str:
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self._alphabeta(gameState, 0, [-float('inf'), float('inf')])[1]

    def _alphabeta(self, gameState: GameState, idx: int, ab: List[float]) -> Tuple[float, str]:
        n = gameState.getNumAgents()

        if idx / n >= self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

        agent = idx % n
        legalActions = gameState.getLegalActions(agent)

        pacman = (agent == 0)
        idx0 = int(pacman)
        idx1 = int(not pacman)
        mod = 1 if pacman else -1
        best_score = -float('inf') * mod
        best_action = None
        for legalAction in legalActions:
            s = gameState.generateSuccessor(agent, legalAction)
            score = self._alphabeta(s, idx + 1, [*ab])[0]
            if score * mod > best_score * mod:
                best_score, best_action = score, legalAction
            if best_score * mod > ab[idx0] * mod:
                break
            ab[idx1] = max(ab[idx1] * mod, best_score * mod) * mod
        
        return (best_score, best_action)

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
        return self._expectimax(gameState, 0)[1]

    def _expectimax(self, gameState: GameState, idx: int) -> Tuple[float, str]:
        n = gameState.getNumAgents()

        if idx / n >= self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

        agent = idx % n
        legalActions = gameState.getLegalActions(agent)
        n_actions = len(legalActions)

        ret_score = -float('inf') if agent == 0 else 0
        ret_action = None

        for legalAction in legalActions:
            s = gameState.generateSuccessor(agent, legalAction)
            score = self._expectimax(s, idx + 1)[0]
            if agent != 0:
                ret_score += score / n_actions
            elif score > ret_score:
                ret_score, ret_action = score, legalAction
        
        return (ret_score, ret_action)


Coordinate = Tuple[int, int]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    walls = currentGameState.getWalls()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()

    score = currentGameState.getScore()
    foods_cost = _food_cost(foods.asList(), pos, walls)
    capsules_cost = _food_cost(capsules, pos, walls)

    ghosts_dis, s_ghosts_dis = _ghost_cost(ghosts, pos, walls)

    def d(x):
        if x == 0:
            return float('inf')
        return 9 / (x**2)

    ghosts_cost = sum(map(d, ghosts_dis))
    s_ghosts_cost = sum(map(lambda x: x[0], filter(lambda x: x[0] < x[1], s_ghosts_dis)))

    return score - foods_cost - capsules_cost - s_ghosts_cost - ghosts_cost


def _ghost_cost(_ghosts: List[AgentState], pos: Tuple[int, int], walls: Grid):
    width = walls.width
    height = walls.height

    ghosts = Grid(width, height)
    for i in range(len(_ghosts)):
        x, y = _ghosts[i].getPosition()
        ghosts[int(x)][int(y)] = _ghosts[i].scaredTimer

    queue = util.Queue()
    queue.push((pos, 0))
    visited = set()
    ghosts_dis = []
    s_ghosts_dis = []
    while not queue.isEmpty():
        cur, dis = queue.pop()
        x, y = cur
        if in_range(cur, width, height) and not walls[x][y] and cur not in visited:
            visited.add(cur)
            if ghosts[x][y] != False:
                if ghosts[x][y] <= 0:
                    ghosts_dis.append(dis)
                else:
                    s_ghosts_dis.append((dis, ghosts[x][y]))
                    pass
            for d in DIRECTIONS:
                queue.push(((x + d[0], y + d[1]), dis + 1))
    return ghosts_dis, s_ghosts_dis


def _food_cost(foodsList: List[Coordinate], pacman: Tuple[int, int], walls: Grid):
    to_foods: Dict[Coordinate, Dict[Coordinate, float]] = {}
    for c in foodsList:
        queue = util.Queue()
        queue.push((c, 0))
        distances = {}
        while not queue.isEmpty():
            pos, dis = queue.pop()
            x, y = pos
            if pos in distances:
                continue
            distances[pos] = dis
            for (dx, dy) in DIRECTIONS:
                next_pos = int(x + dx), int(y + dy)
                nextx, nexty = next_pos
                if next_pos not in distances and not walls[nextx][nexty]:
                    queue.push((next_pos, dis + 1))
        to_foods[c] = distances

    pq = util.PriorityQueueWithFunction(lambda x: x[-1])
    pq.push((foodsList[0], 0))
    visited = set()
    food_cost = min(map(lambda food: to_foods[food][pacman], foodsList))
    while not pq.isEmpty():
        position, cost = pq.pop()
        if position in visited:
            continue
        visited.add(position)
        food_cost += cost
        for food in foodsList:
            if food not in visited:
                pq.push((food, to_foods[food][position]))
    return food_cost


# Abbreviation
better = betterEvaluationFunction
