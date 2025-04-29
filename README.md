# Q-learning in a Haunted House
[Short Project Overview](##_Short_Project_Overview) <br/>
[Approach](##_Approach)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Random Agent](##_Random_Agent)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Q-Learning Agent](##_Q-Learning_Agent)<br/>
[Results](##_Results)<br/>
[Conclusion](##_Conclusion)<br/>

## Short Project Overview

My project is set around the 1981 Atari game 'Haunted House'. I created a Q-Learning agent designed to explore and hopefully beat the haunted house. The agent I created was directly influced from a previous agent I had used for the Pac-Man project, and the Blackjack agent from [Gymnasium's documentation](https://gymnasium.farama.org/introduction/train_agent/). Combining these and making them fit the Atari Haunted House enviroment was not as seamless as I initially expected and forced me to expand my knowledge of Reinforcemnt Learning, enviroments and organization. 

Before I started working on my Learning agent, I created a random agent (inspired by Mario example shared in Pizza) in order to test that my enviroment was being properly called.

The next step was to work on my Q-learning agent. I will go into more detals later, but my largest hurdles here were managing the Atari enviroment and having my agent learn Haunted House's reward system. According to current video game standards, the Atari seems simple, but there is still a lot of infomation for an agent to process such as color, sound, multiple dimensions, size, etc. Once I was able to manage the enviroment consitanly working on the Q-Learning funcions was much simpler.

### Anylisis of Haunted House Rules

In order to familarize myself with Haunted House, which I have never played I read the [offical documentation](https://atariage.com/manual_html_page.php?SoftwareLabelID=233) and watched a [video](https://www.youtube.com/watch?v=Q6m7twIkFyo) on gameply and rules. The highlights I will share below. 

The goal of Haunted House is to explore a mansion, find three pieces of an urn, and escape the mansion. You have nine lives, but if you touch the creatures(spiders, bats, and one ghost), you lose one. You must escape before losing all nine. 

The mansion has 24 rooms, 4 floors with six rooms each. These are connected by staircases and doors. Some doors are locked and require finding the master key. 

The enviroment provided by Gymnasium has 18 possible moves. 
<img width="436" alt="HH1" src="https://github.com/user-attachments/assets/08807216-d626-4a82-8842-65bb03048bfc" />


The movements (right, down, etc) function as expected. "Fire" means the agent lights a match that flickers and allows the area around you to become visable. Matches are unlimited, but your score will decrease with the more you use. To pick up an object you need to run into it. You will drop it if you run into a different object. Urn shards combine into one object.

The final score is based on the number of matches used and the number of lives you have left when you leave the mansion. Matches are counted in the bottom left corner and lives in the bottom right. 

Initally after reading this I felt like I may have bit off more than I could chew. Immediately I attempted to calculate the cost of actions, to encorage getting the key to the locked doors, avoid the creatures, only use matches as necessary etc. Then I realized I was prepping for a heuristic agent, similar to the classes first approch with Pac-Man. The cool thing about Reinforcemnt Learning is you don't need to order your agent to run from the ghost, it learns that on its own. This made Haunted House seem much more approchable. 

Another concern I had after familiarizing myself with the gameplay was sound. The maunal states that the game sounds are very important to the game. I had no clue where to start analzing sound in Gymnnasium, but upon closer anylsis, I realized the sounds provided more of an immersive atmosphere for a human player but are never relied upon to solve puzzles. 

My largest takeaway, outside of learning how to play Haunted House, was the realiztion that the programmer doesn't always need to study the complete makeup of a game exhaustively in order to create an agent that uses Reinforcement Learning to play a game. More important is creating an agent that plays nice with your game's enviroment and understanding the "reward" system of the game. 

## Approach

### Random Agent

The first thing I did was read all of the Gymnasium Documentation. From here I was able to tweak the code provided to set up an inital enviroment that showed my Haunted House. I used the Arcade Learning Environment, [ale](https://ale.farama.org/index.html).

    import gymnasium as gym 
    import ale_py

    gym.register_envs(ale_py)

    #Create Haunted House Env
    env = gym.make("ALE/HauntedHouse-v5", render_mode="human") 
    ...
    

I tested it with an agent that made a random move 1000 times and the enviroment functioned well. I also added a section that would run every possible action 100 time in quick secession while printing the action so that I could familiarize myself with the action movements. 

Video below is sped up.

![HH2](https://github.com/user-attachments/assets/fed1a125-b79e-4b02-bbf1-84a7a5d41598)

As expected the scores returned were typtically negative, all lives were lost, no shards of urn were found and more matches were used then was necessary. In fact, for all of the random agents I ran about half of them never left the first room, and the other half never found a third. 

### Q-Learning Agent

My inital approach to creating a Q-learning project was to take insperation from the previous Pac-Man code I had written and use the example Blackjack agent from Gymnasium to see how it would fit within an Gymnasium game enviroment. The logic behind this is by taking somthing familar that I completely understand I can make the un-familiar more approchable and learn it quicker. Below, I have boiled down this agent to three main issues I ran into and my approches to solving them.

This video is sped up and shows that the agent seems to explore better than the random agent but is still having difficulties.
![hh3](https://github.com/user-attachments/assets/da01ecf1-bb2e-4a5c-a906-fc259349e09e)


#### Hurdle 1: Enviroment
The cornerstone aspects that I had relied upon in the Pac-Man project were not handed to me the same way in the Atari env. I needed to find a way to get the "state" and the "action" from the complex enviroment. I already managed to call it, I needed to find a way to process it. After much trial and error and research, [this article](https://medium.com/@jakemazurkiewicz6/deep-q-learning-ai-to-master-atari-games-e5d2c7862704) by Jake Mazurkiewi was not only applicable to my current challenge but led me to the wrapper's provided by [Gymnasium](https://gymnasium.farama.org/api/wrappers/#module-gymnasium.wrappers). The wrappers that I ended up using were as follows: GrayscaleObservation, ResizeObservation, FlattenObservation,TransformObservation and TimeLimit. These ended up doing all of the heavy lifting for me and created a one dimensional, small, grayscale enviroment for my agent to understand. The inital code I had written to create the enviroment (as I shared above) has doubled so I put in its own file, HH_Env.py 

I liked how clean that looked so I also split up the learning agent and the training/runing functions into their own respective files, Qlearning_agent.py and runQ.py.

The wrappers allowed me to get the state from env.reset() and the actions were gotten from the env using.action_space.n. With these two corner stones I was able to create a learning agent that functioned in the Haunted House Atari enviroment. 

#### Hurdle 2: Time

The next issues I had was the time cost. It is known that it can take a long time to train a model. Yet, I was eager to see how my agent was doing and the lack of feedback after running for a few hours was dishearting. I took another look and found three areas of improvement. 

The largest was changing the render to none, I would no longer get to see my agent explore but this greatly cut down on time. After that I found a wrapper (TimeLimit) that I implemented and allowed me to stop the episode if a number of steps were reached, the goal being to stop running if my agent was stuck. Finally, I changed my approach from printing every episode to saving the results to a txt file. and only printing every ten. 

#### Hurdle 3: Little to No Improvement

Saving the results to a txt file showed me that my average reward was not improving. My first thought was that I set the TimeLimit too low. If I did not give my agent enough time to explore and escape the Haunted House, it would never be able score above zero. I also messed around with the learning_rate, decay and discount to see how they affected my agent. 

I also noted that the score was always negative. That is how the game works, use a match = lose points, lose a life = lose points. There is nothing built that allows you to gain points and that could be negativly impacking my agents ability to learn. I attempted to fix by adding positive rewards for exploration. This did not do as desired so I removed it.

It was while problem shooting the lack of improvement could possibly be sovled by using deep Q-learning instead of Q-learning. I had started with regular Q-learning since I had expereice without and after spending a significat amount of time polishing it, I went back to watch some class lecture videos and found [this article](https://www.baeldung.com/cs/q-learning-vs-deep-q-learning-vs-deep-q-network#:~:text=One%20of%20the%20main%20drawbacks,to%20store%20the%20Q%2Dvalues.), the takeaway I got from the class videos and article is summed up with a direct qoute from the article:

"One of the main drawbacks of Q-learning is that it becomes infeasible when dealing with large state spaces, as the size of the Q-table grows exponentially with the number of states and actions. In such cases, the algorithm becomes computationally expensive and requires a lot of memory to store the Q-values."

Deepmind used Deep neural networks in their [original atari agents ](https://www.youtube.com/watch?v=EfGD2qveGdQ&t=12s), in retrospect I wish I initally started with Deep Q-learning. It is important to note, that it would potentally not have solved this issue, since Haunted House is a delayed reward system with a very large map. I delve more into that in the Results section below.


## Result 
Although I have pride in what I did accomplish in this attempt, I will classify the result as a fail. My learning agent stays stuck in a Haunted House. With little improvement in ability to increase rewards it is doomed to become another ghost haunting the mansion. I have included an example of the rewards.txt in the repository that will highlight the issues with growing (or lack there of with rewards). This is likely the result of many factors. Most prominently that Deep Q-learning would have handled this enviroment better. Other factors inclue the game chosen. It has a very large map, a reward system that focuses on losing points rather than gaining them and due to the gameplay itself, there is no way to score positively unless you find the key, unlock doors, find all urn shards and make your way back out. 

During my anaylsis of this attempt I read [Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals](https://arxiv.org/abs/2302.04449) that discusses the difficultly of using RL to solve atari games with a delayed reward structure, which describes Haunted House. It emphizises the difficulty of this reward structure and proposes a solution that would have the agent "read" the manual prior to attempt. This shows that this choice of game likely influnced these results
## Conclusion
In conclusion, I have learned alot in this attempt and will be making another one. Now that I have a working enviroment and an agent that speaks to it, I must improve the improvemnet factor. 

External factors that negativly affected my agents can be dealt with in the next attempt. It is possible that my impaitene prevented me from allowing the agent to improve on its own. In many of my trials if I saw no improvement after 200 episodes I would change something else and try again. In retrospect there is a (small) chance that some attempts would have eventually had a positive improvement had I let them run.  

Another idea for improvement as explained above would be to change my Q-learning agent to a deep Q-learning agent. Implementing a deep Q-learing agent would take some strain of my computer and potentailly work better with this enviroment. 

These are very real points for improvement within my project. However, this paper I discussed in Results also leads to the conclusion that the reward system in the game I choose had a strong effect on the challenges I faced. I do not regret this choice. Working to this conclusion forced me to analyize what is important and what is vital to working RL agents.

Finally, although the results were not as I had initially hoped, I have learned alot about RL and different enviroments. Most importantly I have a better understanding about what parts of gameplay  and what elements are important to a sucessful RL.


