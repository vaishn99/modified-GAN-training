# modified-GAN-training

  

## "An Online Learning Approach to Generative Adversarial Networks." - A Review



## Abstract:

Even though GANs can accurately model complicated distributions, they are known to be difficult to train due to instabilities caused by a difficult minimax optimisation problem. In this paper, the author views the problem of training GANs as finding a mixed strategy in a zero-sum game. Building on ideas from online learning, we propose a novel training method named Chekov GAN. There are two main contributions of this paper, one in theory other in practice. On the theory side, the author shows that the algorithm proposed provably converges to equilibrium for semi-shallow GAN architectures. On the practical side, the paper proposes an efficient heuristic guided by the theoretical results, which we apply to commonly used deep GAN architectures.
  
## Basic setting-GAN:

- A brief :

There are two networks named generator and discriminator, which are fully parameterised using neural networks. The generator network aims to generate indistinguishable samples from real samples, where indistinguishability is measured by an additional discriminative network.

- Notation and Terminology : 

	Let us denote the data distribution by $P data(x)$ and the model distribution by $P_u(x)$. A probabilistic discriminator is denoted by $h_v: x \to [0;1]$ and a generator by $G_u: z \to x$. The GAN objective is:
    Put the equation here.
     
- Important observations regarding GAN : 

	Like in VAE[[1]](#1) or other Diffusion models[[2]](#2)., we are not solving any optimisation problem but instead finding a nash equilibrium solution. It can be shown that If I have an optimal discriminative network, training the generator is equivalent to minimising the JS divergence between data and model distributions.

- One way to look at the GAN objective is as a minimisation of f-divergence [[3]](#3) for a particular instantiation for f, which is more of a statistical perspective.
- Another way is from a game-theory side. That is to view GAN's objective as finding a Nash equilibrium of a two-player zero-sum game. The paper utilises this view. Here in the GAN setting, Generator(G) and Discriminator(D) are the players, and whenever one player incurs a loss, it is the same as getting a reward for the other player(zero-sum setting). So two-player -zero-sum is justified. Here in the discussion, we will focus on this perspective. 

  

## The training procedure for Vanilla GAN: 

- The optimal parameter is a PSNE, and we need to compute that point. 
- Goodfellow’s approach-Alternative gradient descent: 

	We need to alternatively update the parameters of both blocks. Here we could use SGD to update. By training the vanilla GAN, I understood that the hyperparameters and the initial parameter values play are essential in determining various aspects such as stability, sample diversity etc. I have given a simple version of the algorithm in the figure :

  

	Algorithm: 

  

## Problems with Standard GAN:(Alternating gradient descent) 

- Even for small games, failures are noted (Salimans et al. [[4]](#4)) 
- Convergence and oscillation issues (Metz et al. [[5]](#5)) 
	* Non-convergence refers to the Stability issue.
	* Oscillation across samples refers to Mode collapse.
Here is a problem, and a lot of papers have come up with a variety of solutions. This paper proposes one such.

# Chekov GAN

I assume the reader is familiar with the definition of the following terms: Two player zero-sum games, MSNE, PSNE and FTRL. Here I will give the definitions for eps-MSNE and some commonly used GAN architectures.

## Recall: important definitions :

### 1. Eps-MSNE :



### 3. GAN classification based on architecture:
	Put the image here.

	But the basic info here.

  

## Theoretical background :

- On the existence of MSNE: We have theoretical guarantees for the existence of an MSNE (Mixed Strategy Nash Equilibrium) Nash et al. [[6]](#6)
- Regret bound for FTRL: For the FTRL algorithm, given the loss functions are convex, we have theoretical results for achieving sub-linear regret. (An important result from online learning literature)

- Computation of MSNE: If the loss and reward functions are convex and concave respectively, and these functions are the utility function for D and G respectively, then Freund and Schapire [[7]](#7)) (1999) proposes that no-regret algorithms can be used to find an approximate MSNE where the epsilon follows $ BigO(1 /div sqrt(T))$ 

  

# Contribution-Theory side :

- For a semi-shallow architecture for GAN, the game structure it induces is a semi-concave game when the appropriate choice of activation function is made. By semi-concave, I mean that the objective function is concave with respect to the discriminator.
- The authors propose an algorithm which builds on top of Schapire's work on the Computation of MSNE. The following is the algorithm to find the MSNE:
		image for the algorithm_1
- In this particular setting(semi-shallow GAN), the authors are able to prove the convergence of the algorithm. Here I will provide a proof sketch: 

The proof makes  use of a theorem due to Schapire, which shows that if both $A_1$ and $A_2$ ensure no-regret then it implies convergence to approximate MNE. Since the game is concave with respect to $P_2$, it is well known that the FTRL version $A_2$  appearing in Thm is a no-regret strategy. The challenge  is therefore to show that $A_1$ is also a no-regret strategy. This is non-trivial, especially for semi-concave games that do not necessarily  have any special structure with respect to the generator.However, the loss sequence received by the generator is not arbitrary but rather it follows a special sequence based on the choices of the discriminator, $\{f_t(\cdot) = M(\cdot,v_t)\}_{t}$.  In the  case of semi-concave games, the sequence of discriminator decisions, $\{v_t\}_{t}$ has a special property which "stabilizes" the loss sequence $\{f_t\}_{t}$, which in turn enables us to  establish no-regret for $A_1$.

- Essentially,the contribution made by this paper is to Convert a problem of finding nash equilibrium to a problem of solving an optimization problem.
- The analysis enables us to get some effcient heuristic while training the standard GAN.

# Contribution - Practical side:

The following assumptions/restrictions are taken while proposing the practical version of the algorithm.
- Use FTRL for both the players.
- Take the gradient step each time instead of finding the argmin or argmax.
- Use a Queue of finite length.
- The latest generator is used for generating new samples.

## Algorithm proposed: 

  

- Algorithm(implementation point of view): 

  

![Screenshot from 2022-11-10 23-57-28](https://user-images.githubusercontent.com/113635391/201581232-a8162235-aa9d-407e-ae4d-1d1d301b1bef.png)

  
  

## Algorithm to update the Queue : 





  

This is the A3 that the proposed algorithm refers to.

  
  
  
  
  

### Note : 

There is an error in the specified algorithm. But from implementation, I came to the conclusion the way in which the queue is updated doesn't have a significant impact, at least for the dataset that I have tested with (CelebA and a mixture of gaussian).

  
## Implementation: 

  

### Note : 

  

The algorithm can be seen as an extension of the standard GAN training procedure. If I instantiate the queue length as one and remove the initialisation step, we could obtain back the Vanilla training procedure.

# Experiments:

- The authors are comparing the performance of the proposed GAN algorithm with respect to the standard GAN's performance in the following datasets:

		- MNIST, CelebA [[8]](#8),multi-modal gaussian.

- The comparison is with respect to the following aspects:

	- Stability during the training.
	- Sample diversity.
	- Mode collapse.
	
- The following is the configuration used for training:
- 
		batch_size = 128
		image_size = 64*64
		number of channels = 3
		latent dimension = 100
		epochs=20
		lr (the parameter in Adam optimiser)= 0.0002
		K=10
		C=0.01
- For the gaussian dataset mixture, use a latent space dimension as 2. 
- I have implemented GAN training using CelebA dataset and generated 100 images using the trained generator. The noise follows Normal distribution. I have given the images generated using standard GAN and the proposed approach. For GAN, the DC GAN architecture is used.
   image_1,
   image_2,
- To analyse the improvement in the modal collapse aspect, I have used multi-modal Gaussian. The heat map of the dataset is given in the figure. Symmetrical architectures are used for the Generator as well as for the Discriminator.
	original image-heat map
	image_normal
	image_FTRL
	

# Learning/Takeaways:
-  There were no implementations available as the reference. So I have decided to build on top of standard GAN architectures, and the code will be available on "paper with code" as well as in the following GitHub repo :
- For the same problem, multiple solutions(Papers) were available. But I couldn't get if there are any connections across the solutions, whether one implies the other.
- There is a mistake in the queue updation algorithm given in the paper. But the organisation of the paper is really good, and it is appreciable.
- Here there are multiple loss terms that contribute to the final loss. We need to see how PyTorch compute the gradient with respect to all these terms for a single update. There are some nuances we need to consider.
- Building up theory-to-back concepts in deep learning is hard. Each time we update the parameters, the entire structure changes, which makes things challenging.

## References

<a id="1">[1]</a>
Dijkstra, E. W. (1968).

Go to statement considered harmful.

Communications of the ACM, 11(3), 147-148.

  

<a id="2">[2]</a>

Dijkstra, E. W. (1968).

Go to statement considered harmful.

Communications of the ACM, 11(3), 147-148.

  

<a id="3">[3]</a>

Dijkstra, E. W. (1968).

Go to statement considered harmful.

Communications of the ACM, 11(3), 147-148.

<a id="4">[4]</a>

Dijkstra, E. W. (1968).

Go to statement considered harmful.

Communications of the ACM, 11(3), 147-148.

 <a id="5">[5]</a>

Dijkstra, E. W. (1968).

Go to statement considered harmful.

Communications of the ACM, 11(3), 147-148.
 
<a id="6">[6]</a>

Dijkstra, E. W. (1968).

Go to statement considered harmful.

Communications of the ACM, 11(3), 147-148.

  
  
  
  

<!-- # Plan:

- Implement the algorithm and comapre it's performance with the standard GAN in terms of:

- stability

- sample diversity

- mode collapse

  

(Since the original architecture is not available in the paper.I'm not sure to what extend I could replicate) -->