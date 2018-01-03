"""
This defines the fantasy football task environment.

"""

import gym
import sys
from gym import spaces
from fantasy_football_auction.auction import Auction, AuctionState, InvalidActionError
from math import exp
from six import StringIO


class FantasyFootballAuctionEnv(gym.Env):
    """
    Fantasy football auction draft, with values for each draftable player pre-determined.

    The agent is always assumed to be the first (owner index 0) owner.

    Attributes:
        :ivar Auction auction: the current auction game state. Read only.
        :ivar list(FantasyFootballAgent) opponents: opponents of owner 0. Read only
        :ivar list(Player) players: the draftable players. Read only.
        :ivar int money: the starting money. Read only.
        :ivar list(RosterSlot) roster: the roster each owner must fill. Read only.
        :ivar float starter_value: the weighting of the start vs the bench during scoring. Read only.
        :ivar boolean done: Whether the current match is done. Read only.
        :ivar Error error: if any error happened internally in the auction, it is stored here. Read only.
        :ivar spaces.MultiDiscrete action_space: action space for this env
        :ivar spaces.MultiDiscrete observation_space: observation space for this env
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, opponents, players, money, roster, starter_value):
        """
        :param list(FantasyFootballAgent) opponents: the other opponents in this game
        :param list(Player) players: list of fantasy football auction Players that will be drafted
        :param int money: how much money each owner in the auction starts with
        :param list(RosterSlot) roster: the slots each owner must fill
        :param starter_value: floating point between 0 and 1 inclusive indicating how heavily the final score should
            be weighted between starter and bench. If 1, for example, bench value will be ignored when calculating
            winners.
        """
        # remove any players who cannot be drafted into any of the available slots
        self.players = [player for player in players if any(slot.accepts(player) for slot in roster)]

        self.opponents = opponents
        self.money = money
        self.roster = roster
        self.starter_value = starter_value

        self.auction = Auction(self.players, len(opponents)+1, money, roster)
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.done = False
        self.error = None
        self.turn_count = 0
        self.reward_range = (0, 1)

    @classmethod
    def action_index(cls, auction, player_index, bid):
        """

        :param auction auction for which the index should be calculated
        :param player_index: index of player in the players list
        :param bid: bid amount
        :return int: index into the flattened action space, representing the
            action of nominating / bidding for the specified player with the specified
            bid amount.
        """
        return player_index * auction.money + bid

    def _action_space(self):
        """
        See README.md for details

         We represent the action space as a 2-dimensional matrix. The row
        represents the player index. The column represents the bid on that
        player - one column for every integer from 0 to the maximum possible
        bid amount.

        We flatten this 2-d matrix into a list to get our one action

        It's flattened, so this is represented as a Discrete

        :return: the action space
        """

        return spaces.Discrete((len(self.players)-1) * self.money)

    def _observation_space(self):
        """
         See README.md for details.

        This returns a multidiscrete with the following dimensions (in this order).

        Say we have p draftable players, m starting money, and
        n owners in the game.
        We have these dimensions (in the order described):
        n dimensions in the range (0 - m): the current max bid for each owner
        1 dimension in the range (0 - m): the current bid to beat for the current nominee
        1 dimension in the range (0 - n): the index of the owner who has the current bid to beat
            for the current nominee
        p dimensions in the range (0 - n+1):
          * each dimension represents a draftable player's 'state'
          * a value of 0 - n-1 indicates that the owner with that index owns that player
          * a value of n indicates that the player is undrafted
          * a value of n+1 indicates that the player is the current nominee

        Each of these is represented in a multidiscrete with the dimensions in the order
        specified above.

        :return: the observation space
        """
        dimensions = []
        # one per owner - max bid
        for i in range(len(self.opponents)+1):
            dimensions.append([0, self.money])

        # bid to beat
        dimensions.append([0, self.money])
        # winning bidder
        dimensions.append([0, len(self.opponents)])

        # player status, one per player
        for i in range(len(self.players)):
            dimensions.append([0, len(self.opponents)+1])

        return spaces.MultiDiscrete(dimensions)

    def _encode_auction(self):
        """

        Array with the value of each dimension:

        We have these dimensions (in the order described):
        n dimensions in the range (0 - m): the current max bid for each owner
        1 dimension in the range (0 - m): the current bid to beat for the current nominee
        1 dimension in the range (0 - n): the index of the owner who has the current bid to beat
            for the current nominee
        p dimensions in the range (0 - n+1):
          * each dimension represents a draftable player's 'state'
          * a value of 0 - n-1 indicates that the owner with that index owns that player
          * a value of n indicates that the player is undrafted
          * a value of n+1 indicates that the player is the current nominee

        :return: the observation space of the current auction
        """

        observation = []

        # one per owner - max bid
        for owner in self.auction.owners:
            observation.append(owner.max_bid())

        # bid to beat
        if self.auction.bid is None:
            observation.append(0)
        else:
            observation.append(self.auction.bid)

        # winning bidder
        if self.auction.winning_owner_index() == -1:
            observation.append(0)
        else:
            observation.append(self.auction.winning_owner_index())

        # player status, one per player
        for i, player in enumerate(self.auction.players):
            owner_idx = self.auction.owner_index_of_player(i)
            if owner_idx != -1:
                observation.append(owner_idx)
            else:
                if player == self.auction.nominee:
                    observation.append(len(self.auction.owners) + 1)
                else:
                    observation.append(len(self.auction.owners))

        return observation

    def _reset(self):
        self.auction = Auction(self.players, len(self.opponents) + 1, self.money, self.roster)
        self.done = False
        for opponent in self.opponents:
            opponent.reset()
        self.turn_count = 0

        return self._encode_auction()

    def _close(self):
        self.owner_idx = None
        self.players = None
        self.num_owners = None
        self.money = None
        self.roster = None
        self.auction = None
        self.action_space = None
        self.observation_space = None
        self.done = None
        self.opponent_policies = None
        self.opponent = None
        self.starter_value = None
        self.opponents = None

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.auction) + '\n')
        if self.error:
            outfile.write('Most recent illegal move was: \n' + str(self.error) + '\n')
        if self.done:
            outfile.write('scores: ' + str(self.auction.scores(self.starter_value)) + '\n')
        outfile.write('turn count: ' + str(self.turn_count) + '\n')
        return outfile

    def _act(self, action, owner_idx):
        """

        :param int action: action to perform, should be a single integer value,
            representing the index in the flattened action space of the action
            to perform.
        :return boolean: true iff action was legal
        """

        # based on the index, figure out the action
        # player index is the row (in the original 2-d matrix)
        player_index = action // self.money
        # bid is the column(in the original 2-d matrix)
        bid = action % self.money

        try:
            # 0 bid means do nothing
            if bid != 0:
                if self.auction.state == AuctionState.NOMINATE and self.auction.turn_index == owner_idx:
                    self.auction.nominate(owner_idx, player_index, bid)
                elif self.auction.state == AuctionState.BID:
                    # has to be a bid for the nominee
                    if player_index == self.auction.nominee_index():
                        self.auction.place_bid(owner_idx, bid)
        except InvalidActionError as err:
            self.error = err
            return False

        return True

    def _step(self, action):
        """

        :param action: action state to choose within action space
        :return tuple:
            (observation, reward, done, info)
            observation is a tuple representing the current observation state within observation space.
            reward is a float representing the reward achieved by the previous action
            done is a boolean indicating whether the task is done and it's time to restart
            info is a diagnostic tool for debugging - we just return a dictionary with auction: Auction object
                and error: any error that was raised by Auction
        """

        self.turn_count += 1

        # If already terminal, then don't do anything
        if self.done:
            return self._encode_auction(), 0., True, {}

        # ignore illegal moves
        self._act(action, 0)

        # All opponents play. Assumes they never play invalid moves
        if not self.auction.state == AuctionState.DONE:
            for i, opponent in enumerate(self.opponents):
                self._act(opponent.act(self.auction, i+1), i+1)

        try:
            self.auction.tick()
        except InvalidActionError as err:
            # invalid action means no nomination was done, so arbitrarily nominate a valid player for their value,
            # starting with the least valuable player
            undrafted_players = sorted(self.auction.undrafted_players, key=lambda player: player.value, reverse=True)
            for player in undrafted_players:
                if self.auction.owners[0].can_buy(player, 1):
                    bid = min(max(player.value, 1), self.auction.owners[0].max_bid())
                    self._act(FantasyFootballAuctionEnv.action_index(self.auction, self.auction.players.index(player),
                                                                  max(1, bid)), 0)
            self.error = err

        # Reward: if nonterminal, then the reward is 0
        if self.auction.state != AuctionState.DONE:
            self.done = False
            return self._encode_auction(), 0., False, {}
        else:
            # We're in a terminal state. Reward is a gradient between 0 and 1 depending on standing
            self.done = True
            # reward player based on how far their score was from the top player
            scores = self.auction.scores(self.starter_value)
            my_score = scores[0]
            reward = (my_score / max(scores))
            # reward is only 1 if they won, 0 otherwise
            reward = 0.
            if my_score == max(scores):
                reward = 1.
            else:
                reward = 0.
            return self._encode_auction(), reward, True, {}
