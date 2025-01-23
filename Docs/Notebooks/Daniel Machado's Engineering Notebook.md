### Entry 10/24/2024 
This is my first entry to the notebook, so I will give a quick overview of what we have done so far with the project. The first week we met with John (our product owner) for about an hour at the union, and he explained to us the whole project, his idea of how the project should go about, and answered any questions that we might have.

  

From then, we decided to start by learning more about Neural Networks, and to try and get everyone in our group in the same page, as there were some people that knew about neural networks and others that didn't. We were researching about theory to understand what we are working with, and researching about the framework we are going to be using, Pytorch.

  

Once the project started, we were still researching about this, and we decided to meet with John on Tuesdays. We started the project a bit disorganized but we are getting it together.

  

I started working on the generation of Neural Networks based on user inputs. However, last week we spoke with John and noticed we were having a very different approach than what he wanted. I spent this whole sprint fixing our wrong approach, and finishing up the creation of the networks.

  

So far, my part can: accept user inputs for the architecture and parameters, generate networks, save networks, train networks, and save them based on some success criteria. I still have to finish up and clean my code for better readability and understanding, but the main part is done. I also need to add later on support for GPU training and training in parallel.

  

### Entry 10/29/2024 
Today I did not work in code, as my section of the code is almost fully done. We met woth John and he helped me with some questions I had.

He explained to us how the Lambda expression for deciding whether a network. The basic idea is explained in the following picture:

![[Pasted image 20241103203642.png]]

We have also decided to start working on the SDD, where I will be helping on some parts.

### Entry 11/4/2024
Today I worked on 0 code (yet). I did a bunch of section of the SDD which are all documented on there. I hope to get some code done today, but will probably work on it tomorrow. Currently completed sections 1.1, 1.2., 1.2.2, 1.2.3 1.4, 1.5.

### Entry 11/5/2024
Today I worked on some more of the SDD, and was not able to meet with my team sadly because I was sick :c

### Entry 11/6/2024
Today I worked on the presentation. I started a powerpoint and simply added everything I have done this far in the project and what I struggled with. I also did some minor changes to other sections.

### Entry 11/10/2024
Today I worked on some code. I was able to make the code more organized, and more fitting for a python package. I also fixed a small bugged that didn't allow you to train a network right after another. Lastly, I added support for GPU training. This supports both Nvidia GPUs and Apple M1.

I still have to implement the better way to do the success criteria**

### Entry 11/12/2024
I changed the success criteria and made it so that the user inputs 2 values: Minimum loss for the network to be considered trained
and the convergence threshold, which is the max difference between the last 2 epochs of training to make sure the model is 
converging

### Entry 11/18
Met with John, and made some changes to the code. I got a TA's help (Walter the GOAT) to make my code cleaner, as I had a lot of repetition. I also started looking over Simon's code to integrate the main class with his, and also with Sharif's Code.

### Entry 11/22
Have barely had any time to work on the project, but I've just been looking over Simon's code and trying to refactor it to make it work with mine. Looks like I will have to change some parts of my code, because the way I am saving the Networks is not the same as he is doing.

### Entry 11/29/2024
Today I worked on the SRS and did requirement for the User Interface. I also did the Dataflow diagrams levels 0, 1 and 2.