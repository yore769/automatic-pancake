"""Solver package."""

from .det_solver import DetSolver

# Task registry: task name -> solver class
TASKS = {
    'detection': DetSolver,
}

__all__ = ['TASKS', 'DetSolver']
