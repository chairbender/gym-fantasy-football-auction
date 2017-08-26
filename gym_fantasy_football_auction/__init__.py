import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FantasyFootballAuction-OneQB',
    entry_point='gym_fantasy_football_auction.envs:OneQB'
)
