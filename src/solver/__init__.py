from .heuristics import RandomSolver, FavoriteSolver


def get_solver(args, _method):
    if _method == 'Random':
        return RandomSolver(args)
    if _method == 'Favorite':
        return FavoriteSolver(args)