# modified-GAN-training

A review of the paper : "An Online Learning Approach to Generative Adversarial Networks" <br />

## Abstract:



## Basic setting-GAN:

- A brief :<br />
There are two networks named generator and discriminator,which are fully paramterised using neural networks
The goal of the generator network is to generate samples that are indistinguishable from real samples, where indistinguishability is measured by an additional discriminative network. 

- Notation and Terminology : <br />

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

<!-- Let us denote the data distribution by $$\pdata(\x)$$ and the model distribution by $p_\u(\x)$. A probabilistic discriminator is denoted by $h_\v: \x \to [0;1]$ and a generator by $G_\u: \z \to \x$. The GAN objective is:
\begin{align}
\min_{\u} \max_{\v}M(\u,\v) &= \frac 12 \E_{\x \sim \pdata} \log h_\v(\x) + \frac 12 \E_{\z \sim p_\z} \log (1-h_\v(G_\u(\z)))~.
\label{eq:GAN_objective}
\end{align} -->



- imprortant observations regarding GAN : <br />
Like in VAE[[1]](#1) or other Diffusion models[[2]](#2).,we are not solving any optmisation problem,but instead finding a nash equilibrium solution.It can be shown that If I have an optimal discriminative network,training the generator is equivalent to minimising the JS divergence between data and model distributions. 
- One way to look at the GAN objective is as a minimisation of f-divergence [[3]](#3) for a particular instantiation for f,which is more of a statistical perspective.
- Another way is from a game-theory side.That is to view ,GAN's objective as finding a Nash equilibrium of a two player zero-sum game.The paper utilises this view.Here in GAN setting  Generator(G)  and Discriminator(D) are the players and whenever one player is incuring a loss,it is same as getting a reward for the other player(zero sum setting).So two player -zero sum is justified.Here in the discussion,we will focus on this perspective. <br />

## Training procedure for Vanilla GAN: <br />

- The optimal parameters is a PSNE and we need to compute that point. <br />
- Goodfellow’s approach-Alternative gradient descent: <br />
  We need to alternatively update the parameters of both the blocks.Here we could use SGD to update.By training the vanilla GAN ,I understood that the hyperparameters and the initial parameter values play are important in determining various aspects such as stability,sample diversity etc.I have given a simple version of the algorithm in figure :<br />

  algorithm: <br />

# Problems with Standard GAN:(Alternating gradient descent) <br />

- Even for small games failures are noted (Salimans et al. [[4]](#4)) <br />   

- Convergence and oscillation issues (Metz et al. [[5]](#5)) <br />

    *  Non-convergence refer to Stability issue.<br />
    *  Oscillation across samples refers to Mode collapse.<br />


Here is a problem,and a lot of papers came with a variety of solution.This is paper proposes one such .<br />

# Chekov GAN

I assume that the reader is familar with the defintion of the following terms:Two player zero sum games,MSNE,PSNE.Here I will give the definitions for eps-MSNE and FTRL.I will also give the definitions for some commonly used GAN architectures.<br />

## Recall : important definitions :<br />

### Eps-MSNE <br />


### FTRL : <br />



Important note : <br />

If loss functions are linear/convex ,then FTRL can provide no-regret. 


### GAN classification based on architecture:

put the image here.
But the basic info here.

## Theoretical background :

- On the existence of MSNE :We have theoretical guarentees for the existance for a MSNE (Mixed Strategy Nash Equilibrium) Nash et al.(1950) <br />
- Regret bound for FTRL : For FTRL algoithm given the loss functions are convex,we have theoretical results for achieving sub linear regret.(An important result from online learning literature)<br />
- Computation of MSNE : If the loss,reward functions are convex and concave respectively and these functions are the utility function for D and G respectively,then Freund and Schapire (1999), proposes that no-regret algorithms can be used to find an approximate MSNE where the error ~ Big_O(1/sqrt(T))<br />

# Contribution-Theory side :

- For a semi-shallow architecture for GAN ,the game structure it induces is a semi-concave game when appropriate choice of activation function is made. By semi-concave I mean that the objective function is conave with respect to the discriminator.
- The authers propose an algorithm which build on top of Schapire (1999) 's work on Computation of MSNE.The following is the algorithm to find the MSNE:<br />
 
 image for the algorithm_1

- In this particular setting(semi-shallow GAN),authers are able to prove the convergence of the algorithm.Here I will provide a proof sketch: <br />
- Essentially,the contribution made by this paper is to Convert a problem of finding nash equilibrium to a problem of solving an optimization problem.<br />
- The analysis enables us to get some effcient heuristic while training the standard GAN.<br />

# Contribution - Practical side:

The following assumptions are taken while proposing the practical version of the algorithm.<br />

- Use FTRL for both the players.<br />
- Take the gradient step each time instead of finding the argmin or argmax.<br />
- Use a Queue of finite length.<br />
- The latest generator is used for generating new samples.<br />

## Algorithm proposed: <br />

- Algorithm(implementation point of view): <br />

![Screenshot from 2022-11-10 23-57-28](https://user-images.githubusercontent.com/113635391/201581232-a8162235-aa9d-407e-ae4d-1d1d301b1bef.png)


## Algorithm to update the Queue : <br />

This is the A3 that the proposed algorithm referring to.





### Note : <br />

There is an error in the specified algorithm.But from implementation I came to a conclusion the way in which the queue is updated doesnt have a significant impact, atleast for the dataset that I have tested with (CelebA and mixture of gaussian).

## Implementation: <br />

### Note : <br />

The algorithm can be seen as an extension of the standard GAN training procedure.If I instantiate the queue length as 1,and remove the initialisation step,we could obtain back the Vanilla training procedure.



# Experiments:
- The authers are comparing the performance of the proposed GAN algorithm with respect to the standard GAN 's performance in the following datasets:
        -MNIST,CelebA,multi-modal gaussian.
- The comparison is with respect to the following aspects:
        - Stability during the training.
        - Sample diversity.
        - Mode collapse.
- The following is the configuration is used for training:



- I have implemented GAN trained using CelebA dataset.I have generated 100 images using the trained generator.The noise follows Normal distribution. I have given the images generated using standard GAN and the proposed approach.For GAN I have used the DC GAN architecture. 

- To analyse the the improvement on the modal collapse aspect ,I have used multi-modal Gaussian.The heat map of the dataset is given in the figure. Symmetrical architeures are used for Generator as well as the for the Discriminator.






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






<!-- # Plan:
- Implement the algorithm and comapre it's performance with the standard GAN in terms of:
    - stability
    - sample diversity
    - mode collapse

    (Since the original architecture is not available in the paper.I'm not sure to what extend I could replicate) -->





