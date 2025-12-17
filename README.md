# Chestnut 

Chess engine ("chess") with a brain of a nut ("nut").


### Todo

- [ ] Self-play algorithm
    - [ ] I need a resnet to give me probabillity vector for the child nodes in MCTS
    with policy and value head (NN to guide the MCTS)


- [ ] read this [link](https://www.reddit.com/r/reinforcementlearning/comments/cc5mv4/how_to_incorporate_neural_networks_into_a_mcts/)

- [tips on how to expand upon supervised learning](https://www.reddit.com/r/baduk/comments/lvakhd/looking_for_deeper_understanding_of_alphazero/)

# Notes

## Monte Carlo Tree Search with Upper Confidence Bound

MCTS is used on games with extremely high branching factor that min-max algorihtms cannot handle.

Reference : [link](https://ai-boson.github.io/mcts/)

### Selection

Keep selecting the best nodes (highest UCT) until the leaf node.

wi/ni + c*sqrt(t)/ni

wi = number of wins after the i-th move
ni = number of simulations after the i-th move
c = exploration parameter (theoretically equal to √2)
t = total number of simulations for the parent node

### Expansion 

When UCT is unable to find the sucessor node, it expands the tree by appending all possible state to the leaf node.

### Simulation

After expansion, it simulates the entire game from the selected node until the end of the game. If nodes are picked randomly, it is called light play out. Else, heavy play out uses heuristics and evaluation functions.

### Backpropagation

When reaching the end of a game, it traverses upwards to the root and increment visit scores for all nodes. Then, it updates win score for each node if the position of that player wins the playout. (past moves in traversed tree)

### Rollout / Playout

...
