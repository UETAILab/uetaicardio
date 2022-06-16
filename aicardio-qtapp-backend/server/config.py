import os


class ProdConfig:
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 3334

    REDIS_DB = 0
    REDIS_URL = os.getenv('REDIS_URL')
    if REDIS_URL is None:
        REDIS_URL='redis://localhost:6379'


class DevConfig:
    DEBUG = True
    HOST = 'localhost'
    PORT = 3334

    REDIS_DB = 1
    REDIS_URL = os.getenv('REDIS_URL')
    if REDIS_URL is None:
        REDIS_URL='redis://localhost:6379'


def get_env_config():
    if os.environ.get('ENV', None) == 'prod':
        config = ProdConfig()
    else:
        print('THIS APP IS IN DEBUG MODE. YOU SHOULD NOT SEE THIS IN PRODUCTION.')
        config = DevConfig()
    return config
