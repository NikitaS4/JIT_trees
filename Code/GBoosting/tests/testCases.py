import numpy as np


class ModelOptionsHelper:
    @staticmethod
    def single_tree(use_JIT=True):
        return {
            "min_bins": 1024,
            "max_bins": 1024,
            "patience": 1,
            "tree_count": 1,
            "tree_depth": 8,
            "learning_rate": 1,
            "reg": 0,
            "es_delta": 0,
            "use_jit": use_JIT,
            "batch_part": 1,
            "feature_fold_size": 1.0,
            "random_batches": False
        }
    
    @staticmethod
    def single_split(use_JIT=True):
        return {
            "min_bins": 256,
            "max_bins": 256,
            "patience": 4,
            "tree_count": 100,
            "tree_depth": 1,
            "learning_rate": 0.2,
            "reg": 0,
            "es_delta": 0,
            "use_jit": use_JIT,
            "batch_part": 1,
            "feature_fold_size": 1.0,
            "random_batches": False
        }
    
    @staticmethod
    def ensemble(use_JIT=True):
        return {
            "min_bins": 16,
            "max_bins": 256,
            "patience": 4,
            "tree_count": 100,
            "tree_depth": 4,
            "learning_rate": 0.2,
            "reg": 2,
            "es_delta": 1e-5,
            "use_jit": use_JIT,
            "batch_part": 0.7,
            "feature_fold_size": 0.6,
            "random_batches": True
        }
    
    @staticmethod
    def hard_model(use_JIT=True):
        return {
            "min_bins": 128,
            "max_bins": 1024,
            "patience": 10,
            "tree_count": 500,
            "tree_depth": 5,
            "learning_rate": 0.1,
            "reg": 1,
            "es_delta": 0,
            "use_jit": use_JIT,
            "batch_part": 1,
            "feature_fold_size": 1.0,
            "random_batches": True
        }


class TargetOptionsHelper:
    @staticmethod
    def linear():
        return {
            "func": lambda x: 2*x + 3,
            "data_border": 5,
            "repr": r"$y = 2x + 3$",
            "short_name": "lin",
            "dim": 1
        }
    
    @staticmethod
    def poly():
        return {
            "func": lambda x: x ** 3 - 2 * (x ** 2) + 3,
            "data_border": 4,
            "repr": r"$y = x^3 - 2x^2 + 3$",
            "short_name": "poly",
            "dim": 1
        }

    @staticmethod
    def cos():
        return {
            "func": lambda x: np.cos(x),
            "data_border": 3,
            "repr": r"$y = cos(x)$",
            "short_name": "cos",
            "dim": 1
        }
    
    @staticmethod
    def sin():
        return {
            "func": lambda x: np.sin(x),
            "data_border": 3,
            "repr": r"$y = sin(x)$",
            "short_name": "sin",
            "dim": 1
        }

    @staticmethod
    def squared_form():
        return {
            "func": lambda x, y: x ** 2 + y ** 2,
            "data_border": 1,
            "repr": r"$f(x, y) = x^2 + y^2$",
            "short_name": "sq_form",
            "dim": 2 
        }
    
    @staticmethod
    def saddle():
        return {
            "func": lambda x, y: x ** 2 - y ** 2,
            "data_border": 1,
            "repr": r"$f(x, y) = x^2 - y^2$",
            "short_name": "saddle",
            "dim": 2
        }

    @staticmethod
    def cos_poly():
        return {
            "func": lambda x, y: np.cos(x) + y ** 3 - 2 * y ** 2 + 3,
            "data_border": 1,
            "repr": r"$f(x, y) = cos(x) + y^3 - 2y^2 + 3$",
            "short_name": "saddle",
            "dim": 2
        }
