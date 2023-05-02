from ray import tune


def get_tuning_params(algo: str):
    tuning_params = {
        "lr": tune.uniform(5e-5, 5e-3),
        "gamma": tune.uniform(0.5, 0.99),
    }

    if algo == "DQN":
        tuning_params["n_step"] = tune.randint(1, 10)

    if algo == "A2C":
        tuning_params["lambda"] = tune.uniform(0.5, 1)

    if algo == "PPO":
        tuning_params["lambda"] = tune.uniform(0.5, 1)

    if algo == "SAC":
        tuning_params["n_step"] = tune.randint(1, 10)

    return tuning_params
