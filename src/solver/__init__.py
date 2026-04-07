"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .det_solver import DetSolver

TASKS = {
    'detection': DetSolver,
}

__all__ = ['DetSolver', 'TASKS']
