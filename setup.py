from setuptools import setup

setup(name='gym_fantasy_football_auction',
      version='1.07',
      description='Gym environment for doing a fantasy football auction. A set of players, each with some'
                  ' predetermined value (this is usually based on projections / expert consensus) is available'
                  ' for auction drafting, and the agent must try to auction in such a way as to create the '
                  ' team with the highest value. These environments all simulate other players as simple scripted'
                  ' agents who behave pretty much like the average fantasy player.',
      author='Kyle Hipke',
      author_email='kwhipke1@gmail.com',
      url='https://github.com/chairbender/gym-fantasy-football-auction',
      download_url='https://github.com/chairbender/gym-fantasy-football-auction/archive/1.07.tar.gz',
      install_requires=['gym>=0.7.4,<0.9.6', 'fantasy_football_auction>=0.9.97', 'six>=1.11.0', 'numpy>=1.13.1'],
      keywords=['AI', 'football', 'fantasy', 'auction', 'gym', 'environment'],
      packages=['gym_fantasy_football_auction', 'gym_fantasy_football_auction.envs'],
      package_data={'gym_fantasy_football_auction.envs': ['data/*.csv']},
      )
