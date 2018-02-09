"""
This defines the fantasy football task environment.

"""
from functools import reduce

import gym
import sys
import numpy as np

from fantasy_football_auction.position import Position
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
        :ivar int turn_count: number of turns that have transpired for a given game
        :ivar Error error: if any error happened internally in the auction, it is stored here. Read only.
        :ivar spaces.MultiDiscrete action_space: action space for this env
        :ivar spaces.MultiDiscrete observation_space: observation space for this env
        :ivar tuple(float,float) reward_range: range of the reward (0,1)
        :ivar float final_reward: reward for the game to the agent. zero until the game ends.
        :ivar str reward_function: see __init__'s reward_function documentation
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, opponents, players, money, roster, starter_value, reward_function='3.1'):
        """
        :param list(FantasyFootballAgent) opponents: the other opponents in this game
        :param list(Player) players: list of fantasy football auction Players that will be drafted
        :param int money: how much money each owner in the auction starts with
        :param list(RosterSlot) roster: the slots each owner must fill
        :param starter_value: floating point between 0 and 1 inclusive indicating how heavily the final score should
            be weighted between starter and bench. If 1, for example, bench value will be ignored when calculating
            winners.
        :param str reward_function: optional. option for which reward function to use. Possible values are:
            1 - reward at end of game based on ratio of my_score / max(scores). +1 added for winning (for a total of 2).
                0 during game
            2 - reward at end of game based on standing - evenly distributed between 1 and -1 for first and
             last place. 0 otherwise.
            3 - reward with player_value every time player is acquired, punish with player_value / num_opponents every time
                   another owner gets a player (make sure to consider bench in value calculation). 0 otherwise.
            #.1 where # is 1-3 - same as 1-3 but we punish the agent with -1 and terminal state on illegal move.
                However, for 1.3, since we need to overcome the punishment of opponents drafting other players,
                we set the punishment to the max player value*number of slots*number of opponents. This ensures
                that the total cumulative punishment will never exceed the possible punishment incurred during the
                game (which would encourage the agent to commit illegal moves to end the game instantly)
            #.2 where # is 1-2 - same as 1.1 and 1.2 but we don't count the rule violation as a terminal state.
                Instead, we just keep playing and let the game make a default, bad choice.
        """
        # remove any players who cannot be drafted into any of the available slots
        self.players = [player for player in players if any(slot.accepts(player) for slot in roster)]

        self.opponents = opponents
        self.money = money
        self.roster = roster
        self.starter_value = starter_value

        self.auction = Auction(self.players, len(opponents)+1, money, roster)
        # save this so it doesn't need to be recalculated
        self._max_bid_overall = FantasyFootballAuctionEnv.max_bid_overall(self.auction)
        self.action_space = self._action_space()
        # represents the action legality for doing nothing - only legal action is 0 bid for player 0
        self._do_nothing = [0] * self.action_space.n
        self._do_nothing[0] = 1
        self.observation_space = self._observation_space()
        self.done = False
        self.error = None
        self.turn_count = 0
        # NOTE: not sure if reward range means the max/min possible total reward for an episode or
        # for a step. Assuming its for the step.
        if reward_function.endswith(".1") or reward_function.endswith(".2"):
            if reward_function.startswith("3"):
                highest_player_value = max(player.value for player in self.players)
                self.reward_range = (-(highest_player_value * len(self.roster) * len(self.opponents)), highest_player_value)
            else:
                self.reward_range = (-1, 1)
        else:
            if reward_function.startswith("3") or reward_function.startswith("4"):
                highest_player_value = max(player.value for player in self.players)
                self.reward_range = (-highest_player_value, highest_player_value)
            else:
                self.reward_range = (-1 if reward_function.startswith("2") else 0, 1)
        self.reward_function = reward_function
        self._previous_values = [0] * (len(self.opponents)+1)
        # we save the binary encodings of each player's owner in here because it's expensive to compute each step
        # there's one binary value per player per owner (1 indicating that owner owns that player)
        # this array is [num_owners x num_players], i.e. it can be indexed by
        # [owner_idx][player_idx]
        self._encoded_players_ownership = np.array([[0] * len(self.players)] * (len(self.opponents)+1))

        # similar - stores whether a player is draftable by an owner.
        # indexed by [owner_idx][player_idx]
        self._encoded_players_draftability = [[1 if owner.can_buy(player, 1) else 0
                                               for player_idx, player in enumerate(self.players)]
                                               for owner_idx, owner in enumerate(self.auction.owners)]


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
        return player_index * (FantasyFootballAuctionEnv.max_bid_overall(auction)+1) + bid

    @classmethod
    def max_bid_overall(cls, auction):
        """

        :param Auction auction for which the max bid should be calculated
        :return int: representing the max possible bid that can be submitted -
            equivalent to the total money - number of roster slots (i.e. have to save 1 dollar per slot
        """
        return auction.money - (len(auction.roster)-1)

    def _action_space(self):
        """
        See README.md for details

         We represent the action space as a 2-dimensional matrix. The row
        represents the player index. The column represents the bid on that
        player - one column for every integer from 0 to the maximum possible
        bid amount.

        We flatten this 2-d matrix into a list to get our one action

        It's flattened, so this is represented as a Discrete.

        An action of bidding 0 on player 0 is, by definition, our "do nothing" action

        :return: the action space
        """

        # they can bid 0 - max_bid_overall, inclusive
        return spaces.Discrete(len(self.players) * (self._max_bid_overall+1))

    def _observation_space(self):
        """
         See README.md for details.


        :return spaces.Tuple: the observation space. A tuple of MultiDiscretes, each
        MultiDiscrete having 200 either binary or integer dimensions.
        """
        num_players = len(self.players)
        num_owners = len(self.opponents)+1
        dimensions = []

        # one per owner - indicating ownership
        for _ in range(num_owners):
            dimensions.append(spaces.MultiDiscrete([[0, 1]] * num_players))
        # one per owner - indicating bid value for the player
        for _ in range(num_owners):
            dimensions.append(spaces.MultiDiscrete([[0, self._max_bid_overall]] * num_players))
        # one per owner - indicating max bid
        for _ in range(num_owners):
            dimensions.append(spaces.MultiDiscrete([[0, self._max_bid_overall]] * num_players))
        # one per owner - indicating whether player is draftable by the owner
        for _ in range(num_owners):
            dimensions.append(spaces.MultiDiscrete([[0, 1]] * num_players))
        # one - set to 1 for all players if it is owner 0's turn to nominate
        dimensions.append(spaces.MultiDiscrete([[0, 1]] * num_players))

        # fixed layers
        # ECT value of player
        dimensions.append(spaces.MultiDiscrete([[0, self._max_bid_overall]] * num_players))
        # one layer, binary, per possible position, in this order
        # QB = 0
        # RB = 1
        # WR = 2
        # TE = 3
        # DST = 4
        # K = 5
        # LB = 6
        # DE = 7
        # DT = 8
        # CB = 9
        # S = 10
        for i in range(11):
            dimensions.append(spaces.MultiDiscrete([[0, 1]] * num_players))

        return spaces.Tuple(dimensions)

    def _encode_status(self, player):
        """

        :param Player player:
        :return:
        """
        pass

    def _encode_auction(self):
        """

        Array with the value of each dimension (see observation_space comments)

        :return: the observation space of the current auction
        """
        num_players = len(self.players)
        num_owners = len(self.opponents) + 1

        observation = [
            # one per owner - indicting ownership
            # this is slow and should be stored in memory rather than calculated each step
            *self._encoded_players_ownership.tolist(),
            # one per owner per player, bid values for the player (only nonzero for current
            # nominee)
            *[[0 if player_idx != self.auction.nominee_index() else self.auction.bids[owner_idx]
                for player_idx in range(num_players)]
                for owner_idx in range(num_owners)],
            # one per owner per player, max bid value (regardless of player)
            *[[owner.max_bid()] * num_players
              for owner in self.auction.owners],
            # one per owner per player - draftability
            # this is slow and should be stored in memory rather than calculated each step
            *self._encoded_players_draftability,
            # one binary layer - all ones if it is nomination time for owner 0 (the agent)
            [1 if self.auction.state == AuctionState.NOMINATE and self.auction.nominee_index() == 0 else 0] * num_players,
            # fixed input layer - player value
            [player.value for player in self.players],
            # fixed input layer - binary - one per possible position - player position.
            *[[1 if player.position == position else 0 for player in self.players] for position in Position]
        ]

        return observation

    def _reset(self):
        self.auction = Auction(self.players, len(self.opponents) + 1, self.money, self.roster)
        self.done = False
        for opponent in self.opponents:
            opponent.reset()
        self.turn_count = 0
        self.error = None

        self._previous_values = [0] * (len(self.opponents) + 1)
        self._encoded_players_ownership = np.array([[0] * len(self.players)] * (len(self.opponents) + 1))
        self._encoded_players_draftability = [[1 if owner.can_buy(player, 1) else 0
                                               for player_idx, player in enumerate(self.players)]
                                              for owner_idx, owner in enumerate(self.auction.owners)]

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



    def action_legality(self):
        """

        :return list(int): array corresponding to the action space,
            where only actions which are legal for the agent are set to 1, all others set to 0
        """

        if self.done or \
           (self.auction.state == AuctionState.NOMINATE and self.auction.turn_index != 0):
            # if it is done or it is nomination but not my turn, only legal action is to do nothing
            return self._do_nothing
        elif self.auction.state == AuctionState.NOMINATE and self.auction.turn_index == 0:
            # if it is my turn to nominate, legal actions are to nominate a player
            # which is not already purchased which this player can buy
            # (i.e. draftable player), for a value less than the current
            # max bid. Cannot nominate for 0.
            max_bid = self.auction.owners[0].max_bid()
            return [1 if self._encoded_players_draftability[0][self.player_index_of_action(action_idx)] == 1 and
                    0 < self.bid_of_action(action_idx) <= max_bid else 0
                    for action_idx in range(self.action_space.n)]
        else:
            # it is bidding time. Legal actions are to submit a bid for the player who is currently up
            # for nomination, where the bid is greater than the current bid to beat but
            # less than the agent's maximum bid, only if the player is draftable. Also legal
            # to submit no bid
            min_bid = self.auction.bid + 1
            if self.auction.owners[0].can_buy(self.auction.nominee, min_bid):
                # can still bid on that player for any amount between max bid and current bid to beat
                max_bid = self.auction.owners[0].max_bid()
                return [1 if self.auction.nominee_index() == self.player_index_of_action(action_idx) and
                        (max_bid >= self.bid_of_action(action_idx) >= min_bid or
                        self.bid_of_action(action_idx) == 0)
                        else 0
                        for action_idx in range(self.action_space.n)]
            else:
                # can't still bid - can't take any action
                return self._do_nothing


    def player_index_of_action(self, action):
        """

        :param action: int representing the action to take from the action space
        :return int: index of the player represented by that action
        """
        return action // (self._max_bid_overall+1)

    def bid_of_action(self, action):
        """

        :param action: int representing the action to take from the action space
        :return int: bid amount represented by that action
        """
        return action % (self._max_bid_overall+1)

    def _debug_action_legality(self):
        # returns a 2d, unflattened array, indexed like [player_idx][money], with each
        # value set to 1 or 0 based on whether it is marked as legal or illegal
        action_legality = self.action_legality()
        return [[action_legality[FantasyFootballAuctionEnv.action_index(self.auction,player_idx,bid)] for bid in range(self._max_bid_overall+1)] for player_idx
                in range(len(self.players))]

    def _act(self, action, owner_idx):
        """

        :param int action: action to perform, should be a single integer value,
            representing the index in the flattened action space of the action
            to perform.
        :return boolean: true iff action was legal
        """

        # based on the index, figure out the action
        # player index is the row (in the original 2-d matrix)
        player_index = self.player_index_of_action(action)
        # bid is the column(in the original 2-d matrix)
        bid = self.bid_of_action(action)

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

        # If already terminal, then don't do anything (but we can return the reward)
        if self.done:
            return self._encode_auction(), self._calculate_reward(), True, {}

        self._act(action, 0)
        if self.error is not None and self.reward_function.endswith(".1"):
            # terminal and punish on illegal move if using a .1 reward function
            return self._encode_auction(), self._calculate_reward(), True, {}


        # All opponents play. Assumes they never play invalid moves
        if not self.auction.state == AuctionState.DONE:
            for i, opponent in enumerate(self.opponents):
                self._act(opponent.act(self.auction, i+1), i+1)

        try:
            # during bid, tick and check if a buy was completed so we can update the player ownership list
            nominee_index = None
            winning_index = None
            if self.auction.state == AuctionState.BID:
                nominee_index = self.auction.nominee_index()
                winning_index = self.auction.winning_owner_index()
            self.auction.tick()
            if self.auction.state != AuctionState.BID and nominee_index is not None:
                # state changed, so a buy was completed. Change ownership in the owner array
                self._encoded_players_ownership[winning_index][nominee_index] = 1
                # change the status of draftability for the purchased player
                for owner_idx in range(len(self.auction.owners)):
                    self._encoded_players_draftability[owner_idx][nominee_index] = 0
                # recalculate draftability for the winning owner
                for player_idx, player in enumerate(self.auction.players):
                    # can draft if it can buy and player is unowned
                    self._encoded_players_draftability[winning_index][player_idx] = \
                        1 if self.auction.owners[winning_index].can_buy(player, 1) \
                        and self.auction.owner_index_of_player(player_idx) == -1 else 0
        except InvalidActionError as err:
            # terminal and punish on illegal move if using a .1 reward function
            if self.reward_function.endswith(".1"):
                self.error = err
                return self._encode_auction(), self._calculate_reward(), True, {}
            else:
                # invalid action means no nomination was done, so arbitrarily nominate a valid player for their value,
                # starting with the least valuable player
                undrafted_players = sorted(self.auction.undrafted_players, key=lambda player: player.value, reverse=True)
                for player in undrafted_players:
                    if self.auction.owners[0].can_buy(player, 1):
                        bid = min(max(player.value, 1), self.auction.owners[0].max_bid())
                        self._act(FantasyFootballAuctionEnv.action_index(self.auction, self.auction.players.index(player),
                                                                      max(1, bid)), 0)
                self.error = err

        # check if done
        if self.auction.state == AuctionState.DONE:
            self.done = True
        return self._encode_auction(), self._calculate_reward(), self.done, {}

    def is_winner(self):
        """

        :return boolean: true iff this episode is done and the agent won
        """
        if self.done:
            scores = self.auction.scores(self.starter_value)
            my_score = scores[0]
            return my_score == max(scores)
        return False

    def _calculate_reward(self):
        if (self.reward_function.endswith(".1") or self.reward_function.endswith(".2")) and self.error is not None:
            # punish on error
            if self.reward_function.startswith("3"):
                # find highest value player and punish based on it * roster slots * opponents
                highest_player_value = max(player.value for player in self.players)
                return -(highest_player_value * len(self.roster) * len(self.opponents))
            else:
                return -1
        if self.reward_function.startswith("1"):
            if self.done:
                scores = self.auction.scores(self.starter_value)
                my_score = scores[0]
                if my_score == max(scores):
                    return 2.  # add a bonus +1 for winning
                else:
                    return my_score / max(scores)
            else:
                return 0.
        elif self.reward_function.startswith("2"):
            if self.done:
                scores = self.auction.scores(self.starter_value)
                order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                my_index = order[0]
                # interpolate linearly between 1 and -1 based on position
                slope = -2 / (len(self.opponents))
                my_score = 1 + slope * my_index

                return my_score
            else:
                return 0.
        elif self.reward_function.startswith("3"):
            # calculate delta in score for this step
            new_values = self.auction.scores(self.starter_value)
            deltas = [new_values[i] - self._previous_values[i] if i == 0 else
                      (self._previous_values[i] - new_values[i]) / len(self.opponents)
                      for i in range(len(new_values))]
            self._previous_values = new_values
            return reduce(lambda x, y: x + y, deltas)
        elif self.reward_function.startswith("4"):
            # calculate delta in score for this step
            new_values = self.auction.scores(self.starter_value)
            deltas = [new_values[i] - self._previous_values[i] if i == 0 else self._previous_values[i] - new_values[i]
                      for i in range(len(new_values))]
            self._previous_values = new_values
            # if the agent won bid, reduce reward based on spending
            # if the agent lost bid, increase reward based on opponent spending
            if deltas[0] > 0:
                return reduce(lambda x, y: x + y, deltas) - self.auction.bid
            else:
                return reduce(lambda x, y: x + y, deltas) + self.auction.bid





