import numpy as np


class ModelOptionsHelper:
    @staticmethod
    def single_tree(use_JIT=True):
        return {
            "bins": 1024,
            "patience": 1,
            "tree_count": 1,
            "tree_depth": 8,
            "learning_rate": 1,
            "es_delta": 0,
            "use_jit": use_JIT
        }
    
    @staticmethod
    def single_split(use_JIT=True):
        return {
            "bins": 256,
            "patience": 4,
            "tree_count": 100,
            "tree_depth": 1,
            "learning_rate": 0.2,
            "es_delta": 0,
            "use_jit": use_JIT
        }
    
    @staticmethod
    def ensemble(use_JIT=True):
        return {
            "bins": 256,
            "patience": 4,
            "tree_count": 100,
            "tree_depth": 4,
            "learning_rate": 0.2,
            "es_delta": 1e-5,
            "use_jit": use_JIT
        }
    
    @staticmethod
    def hard_model(use_JIT=True):
        return {
            "bins": 1024,
            "patience": 10,
            "tree_count": 500,
            "tree_depth": 5,
            "learning_rate": 0.1,
            "es_delta": 0,
            "use_jit": use_JIT
        }


class TargetOptionsHelper:
    @staticmethod
    def linear():
        return {
            "func": lambda x: 2*x + 3,
            "data_border": 5,
            "repr": r"$y = 2x + 3$",
            "short_name": "lin"
        }
    
    @staticmethod
    def poly():
        return {
            "func": lambda x: x ** 3 - 2 * (x ** 2) + 3,
            "data_border": 4,
            "repr": r"$y = x^3 - 2x^2 + 3$",
            "short_name": "poly"
        }

    @staticmethod
    def cos():
        return {
            "func": lambda x: np.cos(x),
            "data_border": 3,
            "repr": r"$y = cos(x)$",
            "short_name": "cos"
        }
    
    @staticmethod
    def sin():
        return {
            "func": lambda x: np.sin(x),
            "data_border": 3,
            "repr": r"$y = sin(x)$",
            "short_name": "sin"
        }
