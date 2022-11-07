# modified-GAN-training

A review of the paper : "An Online Learning Approach to Generative Adversarial Networks" <br />

# Overview:

- One way to look at the GAN objective is as a minimisation of f-divergence (f-divergence paper ,2016) for a particular instantiation for f,which is more of a statistical perspective.Another way is from a game-theory side.That is to view ,GAN's objective as finding a Nash equilibrium of a two player zero-sum game.The paper utilises this view.Here in GAN setting  Generator(G)  and Discriminator(D) are the players and whenever one player is incuring a loss,it is same as getting a reward for the other player(zero sum setting).So two player -zero sum is justified <br />

## Theoretical background :

- We have are theoretical guarentees for the existance for a MSNE (Mixed Strategy Nash Equilibrium) Nash et al.(1950) <br />
- For FTRL algoithm given the loss functions are convex,we have theoretical results for achieving sub linear regret.<br />
- If the loss,reward functions are convex and concave respectively and these functions are the utility function for D and G respectively,then Freund and Schapire (1999), proposes that no-regret algorithms can be used to find an approximate MSNE where the error ~ Big(1/sqrt(T))<br />

# problem with Standard GAN:(Alternating gradient descent)

- Convergence and oscillation issues ()<br />
- Even for small games failures are noted () <br />   

# Contribution :

- For a semi-shallow architecture for GAN ,the game structure it induces is a semi-concave game when appropriate choice of activation function is made. By semi-concave I mean that the objective function is conave with respect to the discriminator.
- hence the authers propose an algorithm which build on top of Schapire (1999) 's work.
- In this particular setting(semi-shallow GAN),authers are able to prove the convergence of the algorithm.
- Essentially,the contribution made by this paper :
        Converted a problem of finding nash equilibrium to a problem of solving an optimization problem.

- Algorithm (Analysis point of view):

# image 1
- Algorithm(implementation point of view):

# image 2


# Plan:
- Implement the algorithm and comapre it's performance with the standard GAN in terms of:
    - stability
    - sample diversity
    - mode collapse

    (Since the original architecture is not available in the paper.I'm not sure to what extend I could replicate)





