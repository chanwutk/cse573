# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from typing import Any, Dict, Generic, Tuple, List, TypeVar
import util

State = TypeVar('State')
class SearchProblem(Generic[State]):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self) -> State:
        """
        Returns the start state for the search problem.
        """
        pass

    def isGoalState(self, state: State) -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        pass

    def getSuccessors(self, state: State) -> List[Tuple[State, str, float]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        pass

    def getCostOfActions(self, actions: List[str]) -> List[float]:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        pass


def tinyMazeSearch(problem: SearchProblem[Any]) -> List[str]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


State = TypeVar('State')
def _build_path(
    parents: Dict[State, Tuple[State, str]],
    state: State
) -> List[str]:
    ans: List[str] = []
    state, direction, *_ = parents[state]
    while state is not None:
        ans.append(direction)
        state, direction, *_ = parents[state]
    return ans[::-1]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


State = TypeVar('State')
def _search(
    problem: SearchProblem[State],
    collection,
    heuristic=nullHeuristic
) -> List[str]:
    start = problem.getStartState()
    collection.push((start, None, None, 0, 0))

    state_table: Dict[State, Tuple[State, str, float]] = {}
    state_table[start] = (None, None, float('inf'))

    def lower_cost(state: State, cost: float):
        return state_table.get(state, (float('inf'),))[-1] > cost

    while not collection.isEmpty():
        state, parent, action, cost, _ = collection.pop()
        if not lower_cost(state, cost):
            continue
        state_table[state] = (parent, action, cost)

        if (problem.isGoalState(state)):
            return _build_path(state_table, state)
        for _state, _action, _cost in problem.getSuccessors(state):
            acc_cost = cost + _cost
            heu_cost = acc_cost + heuristic(_state, problem)
            if lower_cost(_state, acc_cost):
                collection.push((_state, state, _action, acc_cost, heu_cost))


State = TypeVar('State')
def depthFirstSearch(problem: SearchProblem[State]) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return _search(problem, util.Stack())


State = TypeVar('State')
def breadthFirstSearch(problem: SearchProblem[State]) -> List[str]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return _search(problem, util.Queue())


State = TypeVar('State')
def uniformCostSearch(problem: SearchProblem[State]):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return aStarSearch(problem)


State = TypeVar('State')
def aStarSearch(problem: SearchProblem[State], heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    collection = util.PriorityQueueWithFunction(lambda x: x[-1])
    return _search(problem, collection, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
