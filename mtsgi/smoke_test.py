import sys
import pytest


import mtsgi.main
import mtsgi.rl.arguments as arguments


def test_msgi_random():
    """A trivial test of the meta evaluation loop of MSGI-Random agent."""
    args = arguments.get_args([])
    args.env_name = 'toywob'
    print(args)
    mtsgi.main.main(args)


if __name__ == '__main__':
    sys.exit(pytest.main(["-s", "-v"] + sys.argv))
