# modified-GAN-training

A review of the paper : "An Online Learning Approach to Generative Adversarial Networks" <br />

# Basic setting:

- One way to look at the GAN objective is as a minimisation of f-divergence (f-divergence paper ,2016) for a particular instantiation for f,which is more of a statistical perspective.
- Another way is from a game-theory side.That is to view ,GAN's objective as finding a Nash equilibrium of a two player zero-sum game.The paper utilises this view.Here in GAN setting  Generator(G)  and Discriminator(D) are the players and whenever one player is incuring a loss,it is same as getting a reward for the other player(zero sum setting).So two player -zero sum is justified <br />

## Theoretical background :

- We have are theoretical guarentees for the existance for a MSNE (Mixed Strategy Nash Equilibrium) Nash et al.(1950) <br />
- For FTRL algoithm given the loss functions are convex,we have theoretical results for achieving sub linear regret.(An important result from online learning literature)<br />
- If the loss,reward functions are convex and concave respectively and these functions are the utility function for D and G respectively,then Freund and Schapire (1999), proposes that no-regret algorithms can be used to find an approximate MSNE where the error ~ Big(1/sqrt(T))<br />

# problem with Standard GAN:(Alternating gradient descent)

- Convergence and oscillation issues (Metz :2016) <br />
- Even for small games failures are noted (Salimans: 2016) <br />   

# Contribution :

- For a semi-shallow architecture for GAN ,the game structure it induces is a semi-concave game when appropriate choice of activation function is made. By semi-concave I mean that the objective function is conave with respect to the discriminator.

- The authers propose an algorithm which build on top of Schapire (1999) 's work.
- In this particular setting(semi-shallow GAN),authers are able to prove the convergence of the algorithm.

- Essentially,the contribution made by this paper :
        Converted a problem of finding nash equilibrium to a problem of solving an optimization problem.

- The analysis enables us to get some effcient heuristic while training the standard GAN.

- Algorithm (Analysis point of view):
![Screenshot from 2022-11-10 23-57-07](https://user-images.githubusercontent.com/113635391/201581223-b7e50eac-4036-4f01-a4ad-861178655a7f.png)




- Algorithm(implementation point of view):

![Screenshot from 2022-11-10 23-57-28](https://user-images.githubusercontent.com/113635391/201581232-a8162235-aa9d-407e-ae4d-1d1d301b1bef.png)






Important obervation:<br />





- The update rule in the implementation view is the the online mirror descent in online-convex optimisation literature with the L-2 regularizer.





# Experiments:
- The authers are comparing the performance of the proposed GAN algorithm with respect to the standard GAN 's performance in the following datasets:
        -MNIST,CelebA
- The evaluation is with respect to the following aspects:
        - stability during the training.
        - mode collapse



<!-- # Plan:
- Implement the algorithm and comapre it's performance with the standard GAN in terms of:
    - stability
    - sample diversity
    - mode collapse

    (Since the original architecture is not available in the paper.I'm not sure to what extend I could replicate) -->





