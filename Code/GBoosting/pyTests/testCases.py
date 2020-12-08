import numpy as np


class SingleSplitCases:
    def lin_5_case():
        # y = 2x + 3, x in [-5, 5]
        return {
            'test_name': 'lin_5_neg',
            'bins': 32,
            'patience': 10,
            'es_delta': 0,
            'tree_count': 1000, # stop with early stopping
            'learning_rate': 0.1,
            'tree_depth': 1,  # 1 tree == 1 split
            'data_border': 5,  # data from [-5; 5] interval
            'target': lambda x: 2 * x + 3,  # y = 2x + 3
            'target_repr': r"$y = -2x + 3$",
            'train_cnt': 10000,
            'valid_cnt': 100,
            'plot_sklearn': False
        }

    def lin_5_neg_case():
        # y = 2x + 3, x in [-5, 5]
        return {
            'test_name': 'lin_5',
            'bins': 32,
            'patience': 10,
            'es_delta': 0,
            'tree_count': 1000, # stop with early stopping
            'learning_rate': 0.1,
            'tree_depth': 1,  # 1 tree == 1 split
            'data_border': 5,  # data from [-5; 5] interval
            'target': lambda x: -2 * x + 3,  # y = 2x + 3
            'target_repr': r"$y = 2x + 3$",
            'train_cnt': 10000,
            'valid_cnt': 100,
            'plot_sklearn': False
        }

    def sin_3_case():
        # y = sin(x), x in [-3, 3]
        return {
            'test_name': 'sin_3',
            'bins': 256,
            'patience': 20,
            'es_delta': 0,
            'tree_count': 1000,
            'learning_rate': 0.05,
            'tree_depth': 1,
            'data_border': 3,
            'target': np.sin,
            'target_repr': r"$y = sin(x)$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': True
        }

    def sin_2_case():
        # y = sin(x), x in [-2, 2]
        return {
            'test_name': 'sin_2',
            'bins': 256,
            'patience': 20,
            'es_delta': 0,
            'tree_count': 1000,
            'learning_rate': 0.05,
            'tree_depth': 1,
            'data_border': 2,
            'target': np.sin,
            'target_repr': r"$y = sin(x)$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': True
        }

    def cos_2_case():
        # y = cos(x), x in [-2, 2]
        return {
            'test_name': 'cos_2',
            'bins': 256,
            'patience': 5,
            'es_delta': 0,
            'tree_count': 1000,
            'learning_rate': 0.05,
            'tree_depth': 1,
            'data_border': 2,
            'target': np.cos,
            'target_repr': r"$y = cos(x)$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': True
        }

    def poly_3_case():
        # y = x ** 3 - 2x ** 2 + 3
        return {
            'test_name': 'poly_3',
            'bins': 256,
            'patience': 5,
            'es_delta': 1e-3,
            'tree_count': 1000,
            'learning_rate': 0.05,
            'tree_depth': 1,
            'data_border': 2,
            'target': lambda x: x ** 3 - 2 * (x ** 2) + 3,
            'target_repr': r"$y = x^3 - 2x^2 + 3$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': False
        }

    def poly_4_case():
        # y = x ** 3 - 2x ** 2 + 3
        # x in [-4; 4]
        return {
            'test_name': 'poly_4',
            'bins': 256,
            'patience': 5,
            'es_delta': 1e-3,
            'tree_count': 1000,
            'learning_rate': 0.02,
            'tree_depth': 1,
            'data_border': 4,
            'target': lambda x: x ** 3 - 2 * (x ** 2) + 3,
            'target_repr': r"$y = x^3 - 2x^2 + 3$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': False
        }


class SingleTreeCases:
    def poly_single_tree():
        # y = x ** 3 - 2x ** 2 + 3
        # x in [-4; 4]
        return {
            'test_name': 'poly_single',
            'bins': 1024,
            'patience': 2,
            'es_delta': 0,
            'tree_count': 1,
            'learning_rate': 1,
            'tree_depth': 8,
            'data_border': 4,
            'target': lambda x: x ** 3 - 2 * (x ** 2) + 3,
            'target_repr': r"$y = x^3 - 2x^2 + 3$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': True
        }
 
    def linear_single_tree():
        return {
            'test_name': 'lin_5_single',
            'bins': 32,
            'patience': 10,
            'es_delta': 0,
            'tree_count': 1, # stop with early stopping
            'learning_rate': 1,
            'tree_depth': 5,  # 1 tree == 1 split
            'data_border': 5,  # data from [-5; 5] interval
            'target': lambda x: 2 * x + 3,  # y = 2x + 3
            'target_repr': r"$y = -2x + 3$",
            'train_cnt': 10000,
            'valid_cnt': 100,
            'plot_sklearn': False
        }

    def cos_2_single_tree():
        # y = cos(x), x in [-2, 2]
        return {
            'test_name': 'cos_2_single',
            'bins': 256,
            'patience': 5,
            'es_delta': 0,
            'tree_count': 1,
            'learning_rate': 1,
            'tree_depth': 8,
            'data_border': 2,
            'target': np.cos,
            'target_repr': r"$y = cos(x)$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': False
        }


class MultiTreeMultiSplitCases:
    def poly_4():
        # y = x ** 3 - 2x ** 2 + 3
        # x in [-4; 4]
        return {
            'test_name': 'poly_multi',
            'bins': 256,
            'patience': 4,
            'es_delta': 1e-5,
            'tree_count': 1000,
            'learning_rate': 0.1,
            'tree_depth': 8,
            'data_border': 4,
            'target': lambda x: x ** 3 - 2 * (x ** 2) + 3,
            'target_repr': r"$y = x^3 - 2x^2 + 3$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': False
        }
 
    def linear_5():
        return {
            'test_name': 'lin_5_multi',
            'bins': 256,
            'patience': 4,
            'es_delta': 1e-5,
            'tree_count': 1000, # stop with early stopping
            'learning_rate': 0.1,
            'tree_depth': 8,  # 1 tree == 1 split
            'data_border': 5,  # data from [-5; 5] interval
            'target': lambda x: 2 * x + 3,  # y = 2x + 3
            'target_repr': r"$y = -2x + 3$",
            'train_cnt': 10000,
            'valid_cnt': 100,
            'plot_sklearn': False
        }

    def cos_2():
        # y = cos(x), x in [-2, 2]
        return {
            'test_name': 'cos_2_multi',
            'bins': 256,
            'patience': 4,
            'es_delta': 1e-5,
            'tree_count': 1000,
            'learning_rate': 0.1,
            'tree_depth': 8,
            'data_border': 2,
            'target': np.cos,
            'target_repr': r"$y = cos(x)$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': False
        }
    
    def cos_2_full():
        # y = cos(x), x in [-2, 2]
        return {
            'test_name': 'cos_2_multi_full',
            'bins': 256,
            'patience': 1000,
            'es_delta': 0,
            'tree_count': 1000,
            'learning_rate': 0.05,
            'tree_depth': 8,
            'data_border': 2,
            'target': np.cos,
            'target_repr': r"$y = cos(x)$",
            'train_cnt': 10000,
            'valid_cnt': 3000,
            'plot_sklearn': False
        }
