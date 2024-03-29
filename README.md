
# "An Online Learning Approach to GAN"(ICLR 2016) - A Review
## Abstract:

Even though GANs can accurately model complicated distributions, they are known to be difficult to train due to instabilities caused by a difficult minimax optimisation problem. In this paper, the author views the problem of training GANs as finding a mixed strategy in a zero-sum game. Building on ideas from online learning, we propose a novel training method named Chekov GAN. There are two main contributions of this paper, one in theory other in practice. On the theory side, the author shows that the algorithm proposed provably converges to equilibrium for semi-shallow GAN architectures. On the practical side, the paper proposes an efficient heuristic guided by the theoretical results, which we apply to commonly used deep GAN architectures.

## Basic setting-GAN:
- A brief: 
	There are two networks named generator and discriminator, which are fully parameterised using neural networks. The generator network aims to generate indistinguishable samples from real samples, where an additional discriminative network measures indistinguishability.
- Notation and Terminology :
	Let us denote the data distribution by $P data(x)$ and the model distribution by $P_u(x)$. A probabilistic discriminator is denoted by $h_v: x \to [0;1]$ and a generator by $G_u: z \to x$. The GAN objective is:

<center>

<img width="400" height="50" align="center" alt="Screenshot 2022-12-14 at 11 42 07 AM" src="https://user-images.githubusercontent.com/113635391/207520244-fe25acad-5fe8-41bc-b547-e774f3794eb4.png">

</center>

  

- Important observations regarding GAN :
	Like in VAE[[1]](#1) or other Diffusion models[[2]](#2)., we are not solving any optimisation problem but instead finding a nash equilibrium solution. It can be shown that If I have an optimal discriminative network, training the generator is equivalent to minimising the JS divergence between data and model distributions.
- One way to look at the GAN objective is as a minimisation of f-divergence [[3]](#3) for a particular instantiation for f, which is more of a statistical perspective.
- Another way is from a game-theory side. That is to view GAN's objective as finding a Nash equilibrium of a two-player zero-sum game. The paper utilises this view. Here in the GAN setting, Generator(G) and Discriminator(D) are the players, and whenever one player incurs a loss, it is the same as getting a reward for the other player(zero-sum setting). So two-player -zero-sum is justified. Here in the discussion, we will focus on this perspective.

## The training procedure for Vanilla GAN:

- The optimal parameter is a PSNE, and we need to compute that point.
- Goodfellow’s approach-Alternative gradient descent:
	We need to update the parameters of both blocks alternatively. Here we could use SGD to update. By training the vanilla GAN, I understood that the hyperparameters and the initial parameter values are essential in determining various aspects such as stability, sample diversity etc. I have given a simple version of the algorithm in the figure :

  
	<img width="680" alt="Screenshot 2022-12-13 at 1 50 41 AM" src="https://user-images.githubusercontent.com/113635391/207575353-f519b54d-c1a9-454d-a80a-d969ae5d343f.png">

## Problems with Standard GAN:(Alternating gradient descent)

- Even for small games, failures are noted (Salimans et al. [[4]](#4))
- Convergence and oscillation issues (Metz et al. [[5]](#5))
* Non-convergence refers to the Stability issue.
* Oscillation across samples refers to Mode collapse.

Here is a problem, and a lot of papers have come up with a variety of solutions. This paper proposes one such.

  

# Chekhov GAN

I assume the reader is familiar with the definition of the following terms: Two zero-sum player games, MSNE, PSNE and FTRL. Here I will give the definitions for eps-MSNE and some commonly used GAN architectures.

## Recall: important definitions :

### 1. Eps-MSNE :

  
<img width="676" alt="Screenshot 2022-12-13 at 1 48 35 AM" src="https://user-images.githubusercontent.com/113635391/207537160-4390c84f-2cdf-43ce-805d-64b9cfc93444.png">
				
				Previously defined notations are followed.
### 2. GAN classification based on architecture:

<img width="703" alt="Screenshot 2022-12-13 at 1 58 09 AM" src="https://user-images.githubusercontent.com/113635391/207520622-c18c9dc9-cd09-4a2b-9c31-e296f7c2ff42.png">

	  Here (a),(b,) and (c) refer to shallow GAN, Semi-shallow GAN and Deep GAN. 
       Respectively.
       
## Theoretical background :

- On the existence of MSNE: We have theoretical guarantees for the existence of an MSNE (Mixed Strategy Nash Equilibrium) Nash et al. [[6]](#6)
- Regret bound for FTRL: For the FTRL algorithm, given the loss functions are convex, we have theoretical results for achieving sub-linear regret. (An important result from online learning literature)
- Computation of MSNE: If the loss and reward functions are convex and concave respectively, and these functions are the utility function for D and G respectively, then Freund and Schapire [[7]](#7)) (1999) proposes that no-regret algorithms can be used to find an approximate MSNE where the epsilon follows $ BigO(1 /div sqrt(T))$

# Contribution-Theory side :

- For a semi-shallow architecture for GAN, the game structure it induces is a semi-concave game when the appropriate choice of activation function is made. By semi-concave, I mean that the objective function is concave with respect to the discriminator.
- The authors propose an algorithm which builds on top of Schapire's work on the Computation of MSNE. The following is the algorithm to find the MSNE:<img width="685" alt="Screenshot 2022-12-13 at 2 35 46 AM" src="https://user-images.githubusercontent.com/113635391/207521132-8038ad29-10b8-4092-83d4-a49f0f4cd7eb.png">

- In this particular setting(semi-shallow GAN), the authors are able to prove the convergence of the algorithm. Here I will provide a proof sketch:
	The proof makes use of a theorem due to Schapire, which shows that if both $A_1$ and $A_2$ ensure no-regret then it implies convergence to approximate MNE. Since the game is concave with respect to $P_2$, it is well known that the FTRL version $A_2$ appearing in Thm is a no-regret strategy. The challenge is therefore to show that $A_1$ is also a no-regret strategy. This is non-trivial, especially for semi-concave games that do not necessarily have any special structure with respect to the generator.However, the loss sequence received by the generator is not arbitrary but rather it follows a special sequence based on the choices of the discriminator, $\{f_t(\cdot) = M(\cdot,v_t)\}_{t}$. In the case of semi-concave games, the sequence of discriminator decisions, $\{v_t\}_{t}$ has a special property which "stabilizes" the loss sequence $\{f_t\}_{t}$, which in turn enables us to establish no-regret for $A_1$.
- Essentially,the contribution made by this paper is to Convert a problem of finding nash equilibrium to a problem of solving an optimization problem.
- The analysis enables us to get some effcient heuristic while training the standard GAN.

# Contribution - Practical side:

The following assumptions/restrictions are taken while proposing the practical version of the algorithm.
- Use FTRL for both the players.
- Take the gradient step each time instead of finding the argmin or argmax.
- Use a Queue of finite length.
- The latest generator is used for generating new samples.

## Algorithm proposed:

- Algorithm-2(Implementable):

![Screenshot from 2022-11-10 23-57-28](https://user-images.githubusercontent.com/113635391/201581232-a8162235-aa9d-407e-ae4d-1d1d301b1bef.png)

## Algorithm to update the Queue :
  
<center>
<img width="434" alt="Screenshot 2022-12-14 at 11 53 27 AM" src="https://user-images.githubusercontent.com/113635391/207521821-f82640b9-c729-4cc6-ab1f-31ee1fc14454.png">
</center>

	This is the A3 that the proposed algorithm refers to.

### Note :

There is an error in the specified algorithm. But from implementation, I came to the conclusion the way in which the queue is updated doesn't have a significant impact, at least for the dataset that I have tested with (CelebA and a mixture of gaussian).

## Implementation:
- Replace the init step with choosing the optimal parameter for the corresponding regulariser. 
- Two queues are maintained. One for the generator and the other for the discriminator.
- DC-GAN [[8]](#8) architecture is used for implementation.

### Note :
The algorithm can be seen as an extension of the standard GAN training procedure. If I instantiate the queue length as one and remove the initialisation step, we could obtain back the Vanilla training procedure.

# Experiments:

- The authors are comparing the performance of the proposed GAN algorithm with respect to the standard GAN's performance in the following datasets:
	- MNIST, CelebA,multi-modal gaussian.
- The comparison is with respect to the following aspects:
	- Stability during the training.
	- Sample diversity.
	- Mode collapse.
- The following is the configuration used for training:
		batch_size = 128
		image_size = 64,64
		number of channels = 3
		latent dimension = 100
		epochs=20
		lr (the parameter in Adam optimiser)= 0.0002
		K=10
		C=0.01
- For the gaussian dataset mixture, use a latent space dimension as 2.

- I have implemented GAN training using CelebA dataset and generated 100 images using the trained generator. The noise follows Normal distribution. I have given the images generated using standard GAN and the proposed approach. For GAN, the DC GAN architecture is used.

<center>

<p>

<img src="https://user-images.githubusercontent.com/113635391/207524568-d2150269-978e-42e8-8d4e-7e477c5ed842.png" width="400" height="400" />

</p>

  

<p>

<em>Images generated with vanilla GAN</em>

</p>

<p>

<img src="https://user-images.githubusercontent.com/113635391/207524640-0ddc833e-28cd-4227-be21-de5bf0fcd5b1.png" width="400" height="400" />

</p>

<p>

<em>Images generated with chekhov GAN</em>

</p>

</center>

- To analyse the improvement in the modal collapse aspect, I have used multi-modal Gaussian. The heat map of the dataset is given in the figure. Symmetrical architectures are used for the Generator as well as for the Discriminator.

<center>

<p>

<img src="https://user-images.githubusercontent.com/113635391/207527253-7bbe527c-af47-4362-922f-8c317b367b89.png" width="400" height="400" />

</p>

  

<p>

<em>Heatmap for the training data</em>

</p>

<p>

<img src="https://user-images.githubusercontent.com/113635391/207527285-f63babf9-4937-46b1-a613-a2a5744c7200.png" width="400" height="400" />

</p>

<p>

<em>Heat map for the generated samples with vanilla GAN</em>

</p>

<p>

<img src="https://user-images.githubusercontent.com/113635391/207527337-18706775-e812-4d92-a00d-b69e8bfd188e.png" width="400" height="400" />

</p>

<p>

<em>Heat map for the generated samples with Chekhov GAN</em>

</p>

</center>

# Learning/Takeaways:

- There were no implementations available as the reference. So I have decided to build on top of standard GAN architectures, and the code will be available on "paper with code" as well as in the following GitHub repo: https://github.com/vaishn99/modified-GAN-training
- For the same problem, multiple solutions(Papers) were available. But I couldn't get if there are any connections across the solutions, whether one implies the other.
- There is a mistake in the queue updation algorithm given in the paper. But the organisation of the paper is really good, and it is appreciable.
- Here multiple loss terms contribute to the final loss. We need to see how PyTorch compute the gradient with respect to all these terms for a single update. There are some nuances we need to consider.[[9]](#9)
- Building up theory-to-back concepts in deep learning is hard. Each time we update the parameters, the entire structure changes, which makes things challenging.

## References
<a id="1">[1]</a>   Auto-Encoding Variational Bayes,https://arxiv.org/abs/1312.6114<br />
<a id="2">[2]</a>   Understanding Diffusion Models: A Unified Perspective,https://arxiv.org/abs/2208.11970 <br />
<a id="3">[3]</a>   f-GAN: Training Generative Neural Samplers using Variational Divergence 
        Minimization,https://arxiv.org/abs/1606.00709 <br />
<a id="4">[4]</a>   Improved Techniques for Training GANs,https://arxiv.org/abs/1606.03498 <br />
<a id="5">[5]</a>    Unrolled Generative Adversarial Networks,https://arxiv.org/abs/1611.02163 <br />
<a id="6">[6]</a>   Lecture 5: Existence of a Nash Equilibrium, MIT lecture series. <br />
<a id="7">[7]</a>    On Learning Algorithms for Nash Equilibria,https://people.csail.mit.edu/costis/learning.pdf <br />
<a id="8">[8]</a>   Unsupervised Representation Learning with Deep Convolutional Generative Adversarial 
       Networks,https://arxiv.org/abs/1511.06434 <br />
<a id="9">[9]</a>	How to combine multiple criteria for a loss function? ,https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348


  

  




  



  


  

