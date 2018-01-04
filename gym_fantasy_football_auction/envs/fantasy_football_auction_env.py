"""
This defines the fantasy football task environment.

"""
from functools import reduce

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
        :ivar int turn_count: number of turns that have transpired for a given game
        :ivar Error error: if any error happened internally in the auction, it is stored here. Read only.
        :ivar spaces.MultiDiscrete action_space: action space for this env
        :ivar spaces.MultiDiscrete observation_space: observation space for this env
        :ivar tuple(float,float) reward_range: range of the reward (0,1)
        :ivar float final_reward: reward for the game to the agent. zero until the game ends.
        :ivar str reward_function: option for which reward function to use. Possible values are:
            1 - reward at end of game based on ratio of my_score / max(scores). 0 during game
            2 - reward at end of game - 1 for victory. 0 otherwise.
            3 - reward with player_value every time player is acquired, punish with player_value every time
                   another owner gets a player (make sure to consider bench in value calculation). 0 otherwise.
            #.1 where # is 1-3 - same as 1-3 but we punish the agent with -1 and terminal state on illegal move.
                However, for 1.3, since we need to overcome the punishment of opponents drafting other players,
                we set the punishment to the max player value*number of slots*number of opponents. This ensures
                that the total cumulative punishment will never exceed the possible punishment incurred during the
                game (which would encourage the agent to commit illegal moves to end the game instantly)
            I considered having -1 when nothing is gained/lost rather than 0, but it doesn't really make
                sense for FF. There's not really anything wrong with taking awhile. The game will progress
                at a certain rate regardless of the agent's behavior. There's not really a way to stall.
            Also considered constantly rewarding with the player's team value*3 - opponent's team value.
                But this doesn't really make sense in fantasy - this would mean the agent wants to prevent
                    lots of bidding wars when they are currently behind. But there's nothing wrong with being
                    behind and letting other players waste money on outbidding each other.
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
            1 - reward at end of game based on ratio of my_score / max(scores). 0 during game
            2 - reward at end of game - 1 for victory. 0 otherwise.
            3 - reward with player_value every time player is acquired, punish with player_value every time
                   another owner gets a player (make sure to consider bench in value calculation). 0 otherwise.
            #.1 where # is 1-3 - same as 1-3 but we punish the agent with -1 and terminal state on illegal move.
                However, for 1.3, since we need to overcome the punishment of opponents drafting other players,
                we set the punishment to the max player value*number of slots*number of opponents. This ensures
                that the total cumulative punishment will never exceed the possible punishment incurred during the
                game (which would encourage the agent to commit illegal moves to end the game instantly)
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
        # NOTE: not sure if reward range means the max/min possible total reward for an episode or
        # for a step. Assuming its for the step.
        if reward_function.endswith(".1"):
            if reward_function.startswith("3"):
                highest_player_value = max(player.value for player in self.players)
                self.reward_range = (-(highest_player_value * len(self.roster) * len(self.opponents)), highest_player_value)
            else:
                self.reward_range = (-1, 1)
        else:
            if reward_function.startswith("3"):
                highest_player_value = max(player.value for player in self.players)
                self.reward_range = (-highest_player_value, highest_player_value)
            else:
                self.reward_range = (0,1)
        self.reward_function = reward_function
        self._previous_values = [0] * (len(self.opponents)+1)
        # we save the binary encodings of each player's owner in here because it's expensive to compute each step
        # there's one binary value per player per owner (1 indicating that owner owns that player)
        self._encoded_players_ownership = [0] * (len(self.opponents)+1) * len(self.players)

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
        # one dimension per owner, representing their current maximum bid.
        # the bid to beat for the current nominee
        # 1 binary dimension per owner representing the current bid winner
        # 1 binary for each owner for each player, representing who owns the player
        # player nomination status (1 binary per player)
        Each of these is represented in a multidiscrete with the dimensions in the order
        specified above.

        :return: the observation space
        """
        dimensions = [
            # one per owner - max bid 2
            *[[0, self.money] for _ in range(len(self.opponents)+1)],
            # bid to beat for current nominee +1 = 3
            [0, self.money],
            # winning bidder (1 binary per owner) +2 = 5
            *[[0, 1] for _ in range(len(self.opponents)+1)],
            # player ownership status, 1 binary for each owner for each player +24=29
            *[[0, 1] for _ in range((len(self.opponents)+1)*len(self.players))],
            # player nomination status (1 binary per player) +2=31
            *[[0, 1] for _ in range(len(self.players))]
        ]
        return spaces.MultiDiscrete(dimensions)

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

        observation = [
            # one per owner - max bid
            *[owner.max_bid() for owner in self.auction.owners],
            # bid to beat for current nominee
            0 if self.auction.bid is None else self.auction.bid,
            # winning bidder (1 binary per owner)
            *[1 if i == self.auction.winning_owner_index() else 0 for i in range(len(self.opponents)+1)],
            # player ownership status, 1 binary for each owner for each player
            # this is slow and should be stored in memory rather than calculated each step
            *self._encoded_players_ownership,
            # player nomination status (1 binary per player)
            *[1 if player == self.auction.nominee else 0 for player in self.auction.players]
        ]

        return observation

    def _reset(self):
        self.auction = Auction(self.players, len(self.opponents) + 1, self.money, self.roster)
        self.done = False
        for opponent in self.opponents:
            opponent.reset()
        self.turn_count = 0
        self.error = None

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
                self._encoded_players_ownership[nominee_index*(len(self.opponents)+1) + winning_index] = 1
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
        if self.reward_function.endswith(".1") and self.error is not None:
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
                return my_score / max(scores)
            else:
                return 0.
        elif self.reward_function.startswith("2"):
            if self.done:
                scores = self.auction.scores(self.starter_value)
                my_score = scores[0]
                return 1 if my_score == max(scores) else 0
            else:
                return 0.
        elif self.reward_function.startswith("3"):
            # calculate delta in score for this step
            new_values = self.auction.scores(self.starter_value)
            deltas = [new_values[i] - self._previous_values[i] if i == 0 else self._previous_values[i] - new_values[i]
                      for i in range(len(new_values))]
            self._previous_values = new_values
            return reduce(lambda x, y: x + y, deltas)



