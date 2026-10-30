
# ------------- Common Parameters --------------
alpha = 0.0167
gamma = 0.99

n = 5
env_id = 2
length_limit = [50.0, 500.0]  # meter; only roads with a length in this range
                              # will have evolving weights over time
seed = 1234


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
