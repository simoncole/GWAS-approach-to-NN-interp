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

### Entry 1/14

I missed the first week of class because I was in DC because of my scholarship. We barely met with John because he got a bit late, but we set up some tasks to be fully defined in the next few day and when we meet with John:

- Add Main branch
- Add all Material into branch
- Delete/ move non - final material into another branch
- Make Branches for testing features
- Make a documents folder in Main
- Move documents to documents folder
- Any features being developed should have their own branch
- Organize Main (We have to figure out how)
- Maybe how John described
- How to move from class to package

### Entry 1/16
We were supposed to meet with John, but he canceled last minute so there was not much we could do as we are still waiting for him to set up what we are going to be working on for the rest of the semester.

### Entry 1/21
Met with John today (finally). He gave us the tasks he wants to have as MVP which are pretty much the following (in order):

PIP package
VEGA
Testing - Unit testing each function
Joint prob

Today we also agreed that I will be the scrum master this semester, and I have set up bi weekly standups to summarize what we have worked on for the past couple of days and punish those who havent done much.

### Entry 1/23
Today we decided who is going to be working on what. I will be doing Unit testing. Honestly didnt do much.

### Entry 1/28
I think we were supposed to meet with John, but it got moved. Today I mostly spoke to Akbas about how I should go about the unit testing. He told me to define the approach of how im choosing the test cases I am going to run. He also told us he wants to see some results - something we could show at the end of our project to summarize the results. I asked John and he basically said:
- Training: The average loss of the test set (using whatever loss function the user specified, e.g. MSE)
- Histograms: Standard Deviation and Mean. You can also provide a fitness value for the curve(s) you fitted to it.

### Entry 1/30
Today we met with john and he formally told us what were the things he wants in the project, but he mostly was talking to Simon. I spoke with John shortly about how to do unit testing, and he told me to take a look at the built-in python class Unittest and testcase. So I will be doing some research in the upcopming days

### Entry 2/2
Worked on the SRS and finished the sections we were missing. I did the social Section.

### Entry 2/4
We finished the new parts added to the SRS. Today was all about finishing the SRS and the sections about the environment, social impact... etc.

### Entry 2/6
Today I couldnt meet with the team :c

### Entry 2/11
Today I asked akbas about unit testing, and the TA told me I should really keep track of all the test cases I make and document them on the System Test Plan (whack). Not much done, just researching on how the Unittest class works.

### Entry 2/13
Couldnt make it to class today, but I got started with the unit testing. A lot harder than I thought. Using mock objects is kind of weird, just have to do some more tests. So far all im doing is white box testing. Starting to think we will need to refractor this code to have smaller methods easier to test.

### Entry 2/18
Today we didnt meet with John. I spoke with Akbas about unit testing. Not a lot done, but I did keep going and worked on some test cases before class.

#### Entry 2/20
Today we decided that I will be stopping testing for a bit. I need to refractor the code I was working on becasue there are some methods that are way too big, which makes it really hard to test. Luckily its just a couple of methods. I spoke with Leah to see if she will be able to take on the testing of the other parts of the code while I refractor the network generation code. Hopefully it wont take too long - I am aiming rto have it done by monday since it is not really that bad