from setuptools import setup

setup(name='gym_fantasy_football_auction',
      version='0.0.1',
      install_requires=['gym>=0.7.4', 'fantasy_football_auction>=0.9.5', 'six>=1.11.0'],
      packages=['gym_fantasy_football_auction', 'gym_fantasy_football_auction.envs'],
      package_data={'gym_fantasy_football_auction.envs': ['data/*.csv']},
      )
