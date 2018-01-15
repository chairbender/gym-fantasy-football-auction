# gym-fantasy-football-auction
Gym environment for a fantasy football auction.

A fantasy football auction involves a set of owners and
a list of draftable players. Each owner has certain slots
on their roster that they must fill with players who play
in certain positions. Owners take turns nominating a player.
When a player is nominated, every tick, owners can submit bids
which are higher than the current bid. This continues until
nobody submits a new bid for a full tick. Play proceeds until
all owners' rosters have been filled.

For the purposes of this environment, we assume that every player
has an agreed upon value (based upon projections, expert consensus, etc...).
The winner of the auction is the player who ends with the highest total value,
weighting bench slots lower than non-bench slots.

The reward for this task is simply the agent's final score divided by the 
score of the winning player.

# Action Space

The only action an owner takes in an auction is to nominate a player
(giving an initial bid) and to place a bid on a currently nominated player.

Each draftable player (regardless of what has happened in a game) has
an index from 0 to `number_of_draftable_players - 1`.

Think of the action space as a 2-dimensional matrix. The row 
represents the player index. The column represents the bid on that
player - one column for every integer from 0 to the maximum possible
bid amount. 

For example, the action of nominating player_id 23 with a starting
bid of 30 dollars would be represented by the state at row 23
and column 30. 

Similarly, assuming player 45 is currently nominated at 22 dollars,
the action of raising the bid to 25 dollars would be represented by
the state at row 45 and column 23.

Then, our action space is just one simple modification on top of this - 
we flatten the 2-d matrix into a list.

# Observation Space
Let's say we have 200 draftable players,
a starting money amount of 100, and 4 owners.

Our observation space is a stack of multiple length 200 layers (length = num_players).

We have these layers whose values change:
* owner (4 layers - one per owner, indicating who owns the player)
* bid value (4 layers one per owner - 0 if not nominated, otherwise, contains integer value representing the most
    recent bids from all other players)
* max bid (4 layers - one per owner, indicating the max bid this owner can make for any player)
* draftable - 4 layers - 
    one binary layer per owner - set to 1 if the player is draftable by the owner - meaning
    they have space for it in the roster and the player is not already owned  
* nomination - binary where all values are set to 1 on
    all players if it is nomination time for the 
    owner whose perspective we are taking, otherwise 0.    

We have the following fixed input layers, which are always the same:
* value - ECR value of each player, as an integer
* position - one binary layer per possible position - 11 layers total

So the whole state is a 2d array of size (4 + 4 + 4 + 4 + 1 + 1 + 11 = 29 x 200)