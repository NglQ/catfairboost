lightgbm_params = {
    'num_iterations': 100,
    'objective': 'binary',
    'device_type': 'cpu',
    'num_threads': 8,
    'seed': 42,
    'deterministic': 'true'
}

fairgbm_params = {
    'num_iterations': 100,
    'device_type': 'cpu',
    'num_threads': -2,
    'seed': 42,
    'deterministic': 'true'
}
