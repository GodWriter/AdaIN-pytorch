from config import parse_args
from solver import Solver

args = parse_args()
solver = Solver(args)

if __name__ == "__main__":
    module = args.module

    if module == 'test_data':
        solver.test_data()
    else:
        print("No this module!")