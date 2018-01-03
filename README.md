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

The following information is all an agent should need to know
in a given game:
* The current maximum bid that each other owner can make
* The owner of each player (or if they are unowned)
* The current nominated player
* The current bid to beat for the nominated player
* The owner who has the current winning bid for the nominated player

Note that we've excluded:
* The value of each player - this is determined at the start of
    each season by whatever estimation you want to use (I
    generally use FantasyPros).
* The position of each player - this is also a fixed aspect
    of each season.    
* The slots that each owner has filled / has yet to fill.
    This can be deduced from the state. IDK if that is 
    a good reason to exclude it.   
* The current money each owner has left (max bid is the only
    thing that's really relevant) 
* The value that was paid for each purchased player -
    I don't think this is really needed, but
    I could be wrong. It would be useful if we were trying to
    learn to exploit certain player's tendencies, but that's not
    really in scope for this project.
* The bids submitted for each owner for the current nominee, regardless of
    whether they are the maximum. I think this is another thing that would
    only be useful if we are trying to exploit other players' tendencies.
    For an agent, we just need to know what the current bid to beat is and
    who has it.
    
Given that, we'll describe our observation space. Let's say we have 200 draftable players,
a starting money amount of 100, and 4 owners.

Given that, we have these 206 dimensions:
* 4 dimensions in the range (0 - 100): the current max bid for each owner
* 1 dimension in the range (0 - 100): the current bid to beat for the current nominee
* 1 dimension in the range (0 - 3): the index of the owner who has the current bid to beat 
    for the current nominee
* 200 dimensions in the range (0 - 5):
  * each dimension represents a draftable player's 'state'
  * a value of 0 - 3 indicates that the owner with that index owns that player
  * a value of 4 indicates that the player is undrafted
  * a value of 5 indicates that the player is the current nominee
  
More formally - if we have p draftable players, m starting money, and n owners,
we have `n + p + 2` dimensions:
* n dimensions in the range (0 - m): the current max bid for each owner
* 1 dimension in the range (0 - m): the current bid to beat for the current nominee
* 1 dimension in the range (0 - n-1): the index of the owner who has the current bid to beat 
    for the current nominee
* p dimensions in the range (0 - `n+1`):
  * each dimension represents a draftable player's 'state'
  * a value of 0 - `n-1` indicates that the owner with that index owns that player
  * a value of n indicates that the player is undrafted
  * a value of n+1 indicates that the player is the current nominee
  
You probably don't want to use this encoding directly. I used a Keras Embedding layer in order to come up with a more useful representation.
    
           
