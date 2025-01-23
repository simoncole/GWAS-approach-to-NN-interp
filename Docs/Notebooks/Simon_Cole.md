# 10/10/2024
-	Went over current code to find next steps
-	Researched ways to remove jupyter outputs from git (this still needs to be implemented)
-	Modified existing code to accept dynamic config file 
o	https://github.com/sharifware/Large-Scale-Design-and-Analysis-of-Neural-Networks/commits/Config-File
-	Added environment variables implementation
o	https://github.com/sharifware/Large-Scale-Design-and-Analysis-of-Neural-Networks/commits/environment-variables

# 10/13/2024
- Wrote script to generate data for the simple regression function: $f(a, b) = \frac{1}{5} a^2 - \frac{1}{10} b^3$
- saves as csv
- updated inputNetwork.ipynb to:
-  load data as tensor
- split data into train and test
- dynamically instantiate one NN as test
- Added new architecture file, simpleRegArchitecture.py
- modified code to instantiate n number of networks specified in the config file
- added function to train a network and print how the loss is progressing across epochs
- This function is iteratively called for each instantiated network

# 10/20/2024
-	Planned binning of weights
-	Reconfigured github repository to have develop and main branches
-	Modified pipeline to save trained networks as a state dict
-	Loaded saved networks into new file, binningWeights.ipynb
-	A question I have for John: Should the min and max values for the weights be dynamic based on the data where the min is the smallest data point and the max is the largest or should the min and max be a set number specified by the user like 0 and 1? Currently I’ll implement like the min and max are dynamic
-	Wrote functions to generate the bins based on the weights in the loaded network and then populate the counts of weights in those bins.
-	Made simple histogram with this data

# 10/22/2024
- Realized in class today during our meeting with John that I was really misunderstanding how to bin the weights, no wonder it was suspiciously simple
- I was simply aggregating all the weights for all the networks then creating bins and histograms
- I'll have to instead create bins for each weight in the architecture of the networks and populate those bins with data aggregated from all the networks.
- This was the plan the whole time and I knew that, I just wasn't thinking it through all the way earlier.
- This will be more complex
- I've written loops to iterate through and create an empty matrix of zeroes, however I think there's an issue with the shape of it
- I was struggling for a while with creating empty bins of the right shape for all the layers
- The approach I've taken is to gnerate the data for each layer separately then add them to an array

# 10/23/2024
- After a lot of thinking I figured out how to create empty matrices of the correct shape and how to save all the weights in the shape I need
- I've finished the iteration through populating the bins, it's gotten pretty complex but I think it's working
- The code could definitely do for some fresh eyes looking at it becuause it did get pretty complex
- implemented initial histogram generation for each weight but it's not really working yet, need to look at it more in a future sprint, as planned.
- Realized we never actually started the Jira sprint, in the future we should definitely need to be more on top of Jira
- Made branch for the notebooks

# 10/29/2024
- question for John, should the min and max values for the weight bins be calculated as the min and max of all the networks or should they be for each layer?
- wrote funciton to get absolute min and max of all the networks
- realized function I was using to generate bin edges was generating edges based on distribution of the data, not evenly spaced bins from the min and max
- reviewed and cleaned up code from last time
- realized I was getting bug in the binning becuase np.digitize is 1-indexed
- fixed issue where the upper and lower bound bin were rounding and the digitize value was > or < it should have been by adding 1e-6

# 11/5/2024
- question, which is more intuitive way to input to the plot weight bins funtion?
    1. tuple containing the position of the weight
    2. seperate parameters for neuron and incoming weight position
- updated the way the bins are constructed to go from shape(layer, bin, neuron, incoming weight) to (layer, neuron, incoming weight, bin)
- This makes selecting a distribution for a weight much more intuitive.
- working on function to step through the weights
- Had a helpful talk with John today, learned the desired way to step through the weights, should have the following two options:
    for the first fully connected layer, n:
    1. for each neuron view in succession each weight from the previous layer connecting to that neuron before moving to next neuron in the layer
    2. iterate over each neuron in the current layer before moving on to the next from weight in the previous layer
- got answer for previous question, need to modify the min and max to be local to layers
- finished walking through weights

# Between 11/5 and 14
- totally forgot to write in notebook
- mostly polishing binning of weights walkthroughs, meetings, presentation, then working on the fits

# 11/14/2024
- modified code to get local min and max for each layer
- discussed poster with leah 
- initial ideas are one main figure which is a flowchart of input and the major steps of the product
- shows historgrams and linear fits
- poster should also include diagram of NN and its weights
- cleared up a few things about the SDD

# 11/16/2024
- brainstormed new idea for "A GWAS inspired approach to Neural Network Interperatibilty"
- made diagram in actual notebook
- promising work for next semester I think
- reran pipe with 100 bins instead of 10 to see if there was a more exaggerated difference in the distribution
    - maybe there is? it still looks pretty similar with the new architecture
- what is the optimal number of bins? 
    - needs to strike balance between generalization, a sufficient amount of data in each bin, and actually being small enough to detect differences
    - I think the answer lies between 10 and 100 but could scale with the amount of training data so more data = more bins
- normalized bin counts to all equal 1

# 11/17/2024
- continued work trying to get the curve to fit the histograms
- Met with John to discuss an issue where all the distributions look very similar
- after graphing the regression fit, it appears the predicted data isn't fitting well to the curve.
- Could be due to not batching of the data / training process
- these issues should (hopefully) be fixed once we Dani integrates his class for training with the preliminary version I wrote

# 11/19/2024
- Helped Dani integrate his class with the binning weights code in class today
- met with John and tried to train a couple networks using the new system.
- worked on fitting curve to histogram



# 11/27/2024
- commited code I've been working on the past few days of training a new network in the class implementation
- for my reference: networks_state_dict = {}
for i, network in enumerate(networks):
    train_network(X_train, y_train, X_test, y_test, network, num_epochs)
    print(f"finished training network: {i}")
    networks_state_dict[f'network_{i}'] = network.state_dict()

Save the state dict as a file
output_dir = config.get("output_dir")
torch.save(networks_state_dict, f'{output_dir}/trainedNetworks.pt')

### forgot to update my journal in this stretch :(

# 12/3/2024
- pushed code for visualizing and comparing unique distributions so I could retrain a new batch of networks
- something happend to it when I tried to show it to John in class. 
- I guess I'll have to redo it unfortanelty 

# 12/4/2024
- rewrote the code that I somehow lost yesterday and pushed
- still don't know what happened with that pretty frustrating
- I feel like with this we've accomplished the minimum viable product we set out to at the beginning of the semester.
- Trying to get presentation things figured out
- finished recording my part of the presentation, may have to modify after others since we didn't do it in person
