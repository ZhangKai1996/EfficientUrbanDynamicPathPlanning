
# ------------- Common Parameters --------------
alpha = 0.0167
gamma = 0.99

n = 6
env_id = 2
length_limit = [50.0, 500.0]  # meter; only roads with a length in this range
                              # will have evolving weights over time
seed = 1234

# Only Static(n=0)  1.0  0.346185
# Only Static(n=0)  1.0  0.328357
# Only Static(n=0)  1.0  0.329863

# Only Static(n=2)  0.897  1.196883

# Only Static(n=4)  0.872  1.399282
# Only Static(n=4)  0.887  1.248813

# Only Static(n=6)  0.949  0.657705
# Only Static(n=6)  0.955  0.595119
# Only Static(n=6)  0.949  0.653426

# Only Static(n=8)  0.921  0.937896
# Only Static(n=8)  0.921  0.928032
# Only Static(n=8)  0.935  0.79466

# Only Static(n=10)  0.928  0.874665
# Only Static(n=10)  0.922  0.937232
# Only Static(n=10)  0.917  0.983522

# Only Static(n=12)  0.95   0.669736
# Only Static(n=12)  0.946  0.706321
# Only Static(n=12)  0.941  0.762328


def get_kwargs(env_id_, radius_=None):
    env_kwargs_ = {'place_name': None,
                  'network_type': 'drive',
                  'center_idx': 6666,
                  'remove': True,
                  'render': True,
                  'max_step': 100}

    kwargs_ = {'num_episodes_train': 100,
              'epsilon_decay': 0.1,
              'epsilon_end': 0.1,
              'batch_size': 256,
              'save_rate': 100}

    # Nanjing, China
    if env_id_ == 1:
        if radius_ is None: radius_ = 2e4
        env_kwargs_['max_step'] = 200
        env_kwargs_['center_idx'] = 12000
        env_kwargs_['place_name'] = "Nanjing, China"
        kwargs_['num_episodes_train'] = int(1e5)
        return radius_, env_kwargs_, kwargs_

    # Chaoyang, Beijing, China
    if env_id_ == 2:
        if radius_ is None: radius_ = -1
        env_kwargs_['max_step'] = 200
        env_kwargs_['center_idx'] = 8850
        env_kwargs_['place_name'] = "Chaoyang, Beijing, China"
        kwargs_['num_episodes_train'] = int(1e5)
        return radius_, env_kwargs_, kwargs_

    # Pudong, Shanghai, China
    if env_id_ == 3:
        if radius_ is None: radius_ = 1.8e4
        env_kwargs_['max_step'] = 250
        env_kwargs_['place_name'] = "Pudong, Shanghai, China"
        kwargs_['num_episodes_train'] = int(1e5)
        return radius_, env_kwargs_, kwargs_

    raise NotImplementedError


radius, env_kwargs, kwargs = get_kwargs(env_id_=env_id)
