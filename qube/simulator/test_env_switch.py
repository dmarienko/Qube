from qube.configs import Properties

__TESTING_SERVER_ENV = Properties.get_env()


def switch_env(new_env) -> None:
    """
        Method allows to switch testing server. Say from Default to Mars
    """
    global __TESTING_SERVER_ENV
    __TESTING_SERVER_ENV = new_env


def get_env() -> str:
    """
        Shows what's current testing server
    """
    return __TESTING_SERVER_ENV
