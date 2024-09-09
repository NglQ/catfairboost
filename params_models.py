lightgbm_params = {
    'num_iterations': 100,
    'objective': 'binary',
    'device_type': 'cpu',
    'num_threads': 8,
    'seed': 42,
    'deterministic': 'true'
}

fairgbm_params = {
    'multiplier_learning_rate': 0.005,
    'num_iterations': 100,
    'device_type': 'cpu',
    'num_threads': 8,
    'seed': 42,
    'deterministic': 'true'
}

catboost_params = {
    'iterations': 100,
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1
}
