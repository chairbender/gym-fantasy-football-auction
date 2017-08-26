import gym
import sys
from gym import error, spaces, utils
from gym.utils import seeding
from fantasy_football_auction.auction import Auction, AuctionState
from fantasy_football_auction.position import Position, RosterSlot
from six import StringIO

### Adversary policies ###
def make_random_policy(np_random,owner_idx):
    """

    Policy which makes a random choice mostly among valid choices (instead of among the entire state space)
    :param np_random: random seed
    :param owner_idx: id of the owner to play as
    """
    def random_policy(auction):
        me_owner = auction.owners[owner_idx]
        if auction.state == AuctionState.NOMINATE and \
            auction.turn_index == owner_idx:
            # randomly nominate someone that is left with a bid between 1 and max
            chosen_nominee = np_random.choice(me_owner.possible_nominees())
            auction.nominate(owner_idx,chosen_nominee.fid,range(1, me_owner.max_bid()+1))
    return random_policy


class FantasyFootballAuctionEnv(gym.Env):
    """
    Fantasy football auction draft, with values for each player pre-determined.
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, owner_idx, num_owners, players, money, roster, opponent):
        """

        :param owner_idx: index of the agent player, must be in [0,num_opponents)
        :param num_owners: number of owners total in the game (including the agent)
        :param players: list of fantasy football auction Players that will be drafted
        :param money: how much money each owner in the auction starts with
        :param roster: how many slots each owner must fill
        :param opponent: AI to use for all opponents. Possibilities are: 'random' TODO: add a smarter scripted policy
        """
        self.owner_idx = owner_idx
        self.players = players
        self.num_owners = num_owners
        self.money = money
        self.roster = roster

        self.auction = Auction(players, num_owners, money, roster)
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.done = False

        self.opponent_policy = None
        self.opponent = opponent

    def _seed(self, seed=None):
        # Used to seed the random opponent
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1]

    def _action_space(self):
        """
         We model the action space like so:
        - Nomination for all remaining draftable players [0...len(players)] (0 is a noop), index players starting at 1
        - Bid amount [0 - money] (used for nomination starting bid and during bid rounds)

        :return: the action space
        """

        return spaces.MultiDiscrete([[0, len(self.players)], [0, self.money]])

    def _observation_space(self):
        """
         Observation space is a bit more complicated. For referencing players we go by index
        of the players list parameter.

        Each player's position - MultiDiscrete [0-numpositions]*numplayers
        Each player's value - multiDiscrete [0-money] * numplayers
        Each player's owner - multiDiscrete [0-numplayers] * numplayers 0 for no owner 1 for owner 1, etc...
        Each player's sold-for amount (0 for being still draftable) [0-money] * numplayers

        Current nominee [0 - numplayers] (0 for no nominee)

        Each owner's current max bid ability [0 - money] * numowners
        Each owner's current bid [0 - money] * numowners
        roster slot types for each owner

        :return: the observation space
        """
        spaces.Tuple(
            (
                # player position
                spaces.MultiDiscrete([0 - len(Position) - 1] * len(self.players)),
                # player value
                spaces.MultiDiscrete([0 - self.money] * len(self.players)),
                # player owner
                spaces.MultiDiscrete([0 - self.num_owners] * len(self.players)),
                # player's sold for amount
                spaces.MultiDiscrete([0 - self.money] * len(self.players)),
                # current nominee
                spaces.Discrete([0 - len(self.players)]),
                # owner's max bid
                spaces.MultiDiscrete([0 - self.money] * self.num_owners),
                # owner's current bid
                spaces.MultiDiscrete([0 - self.money] * self.num_owners),
                # roster slot types for each slot
                spaces.MultiDiscrete[[0 - len(RosterSlot.slots) - 1] * len(self.roster)]
            )
        )

    def _playerOwnersPurchases(self):
        """

        :return: an array with each element corrresponding to each player in self.players indicating the owner
        of the player and the purchase of that player. None indicates no owner. The element is a dictionary with an owner and a purchase
        """
        result = [None] * len(self.players)
        for owner in self.auction.owners:
            for purchase in owner.purchases:
                player_index = self.players.index(purchase.player)
                result[player_index] = {owner: owner, purchase: purchase}

        return result

    def _encode_auction(self):
        """

        :return: the observation space of the current auction
        """
        player_owners = self._playerOwnersPurchases()
        return (
            # player position
            [player.position.value for player in self.players],
            # player value
            [player.value for player in self.players],
            #player owner
            map(player_owners, lambda entry: entry.owner.id),
            #player sell amount
            map(player_owners, lambda entry: 0 if entry is None else entry.purchase.cost),
            #current nominee
            self.players.find(self.auction.nominee) + 1,
            #owner's max bid
            [owner.max_bid() for owner in self.auction.owners],
            #owner's current bid
            self.auction.bids,
            # roster slot types
            [RosterSlot.slots.find(roster_slot) for roster_slot in self.roster]
        )


    def _reset(self):
        self.auction = Auction(self.players, self.num_owners, self.money, self.roster)
        self.done = False

        return self._encode_auction()

    def _close(self):
        self.opponent_policy = None
        self.state = None

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.auction) + '\n')
        return outfile

    def _act(self, action, owner_idx):
        """

        :param action: action to perform, should be an array with 2 values, the first being the
            player to nominate (0 to not nominate), the second being the bid amount (0 to not bid)
        :param owner_idx: index of owner who should do the action
        :return: true if success, false if illegal move
        """
        if action[0] > 0:
            return self.auction.nominate(owner_idx,self.players[action[0]-1],action[1])
        elif action[1] > 0:
            return self.auction.place_bid(owner_idx,action[1])

    def _step(self, action):
        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}

        if not self._act(action, self.owner_idx):
            # Automatic loss on illegal move
            self.done = True
            return self.state.board.encode(), -1., True, {'state': self.state}

        # All opponents play
        if not self.auction.state == AuctionState.DONE:
            for i in range(0,self.num_owners):
                if i != self.owner_idx:
            self.opponent_policy(self.auction)

        # Reward: if nonterminal, then the reward is 0
        if not self.state.board.is_terminal:
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal
        self.done = True
        white_wins = self.state.board.official_score > 0
        black_wins = self.state.board.official_score < 0
        player_wins = (white_wins and self.player_color == pachi_py.WHITE) or (
        black_wins and self.player_color == pachi_py.BLACK)
        reward = 1. if player_wins else -1. if (white_wins or black_wins) else 0.
        return self.state.board.encode(), reward, True, {'state': self.state}

    @property
    def _state(self):
        return self.state

    def _reset_opponent(self):
        if self.opponent == 'random':
            self.opponent_policy = make_random_policy(self.np_random)
        else:
            raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))