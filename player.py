from datetime import datetime
from my_nn import MyNN
import retro
import tensorflow as tf
import numpy as np
import os
import random
from util import stack_frames
from collections import deque
from memory import Memory
import warnings
import itertools
warnings.filterwarnings('ignore')
import yaml

class Player:
    def __init__(self, game):
        with open("config.yaml", 'r') as stream:
            try:
                config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.batch_size     = config['batch_size']
        self.learning_rate  = config['learning_rate']
        self.memory_size    = config['memory_size']
        self.gamma          = config['gamma']
        self.epsilon        = config['epsilon']
        self.explore_start  = config['explore_start']
        self.explore_stop   = config['explore_stop']
        self.decay_rate     = config['decay_rate']
        self.decay_step     = config['decay_step']
        self.total_episodes = config['total_episodes']
        self.max_steps      = config['max_steps']

        self.env = retro.make(game=game)
        self.memory = Memory(max_size = self.memory_size)

        self.action_size = self.env.action_space.n
        self.state_size = [38, 42, 4]
        self.possible_actions = np.array(np.identity(self.action_size, dtype=int).tolist())
        self.possible_actions = list(itertools.product((0,1), repeat=self.action_size))
        self.action_size = len(self.possible_actions)

        tf.reset_default_graph()
        self.myNN = MyNN(self.action_size, self.state_size, self.learning_rate)

    def init_memory(self):
        state = self.env.reset()
        stacked_frames = deque([np.zeros((38, 42), dtype=np.int) for i in range(4)], maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        for i in range(self.batch_size):
            choice = random.randint(1,len(self.possible_actions))-1
            # action = possible_actions[choice]
            action = np.zeros(512, dtype=np.int)
            action[choice] = 1
            next_state, reward, done, _ = self.env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            self.memory.add((state, action, reward, next_state, done))
            state = next_state

    def train(self, render = False):
        self.init_memory()
        state = self.env.reset()
        stacked_frames = deque([np.zeros((38, 42), dtype=np.int) for i in range(4)], maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            total_rewards = 0
            episode = 0
            for episode in range(self.total_episodes):
                step = 0
                state = self.env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)
                # episode += 1
                while step < self.max_steps:
                    a = datetime.now()
                    exp_exp_tradeoff = np.random.rand()
                    explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)

                    if (explore_probability > exp_exp_tradeoff):
                        choice = random.randint(1,len(self.possible_actions))-1
                        action = self.possible_actions[choice]
                    else:
                        Qs = session.run(self.myNN.output, feed_dict = {self.myNN.input: state.reshape((1, *state.shape))})
                        choice = np.argmax(Qs)
                        action = self.possible_actions[choice]

                    batch = self.memory.sample(self.batch_size)
                    target_Qs_batch = []
                    memory_states = []
                    memory_actions = []
                    memory_rewards = []
                    memory_next_states = []
                    memory_dones = []
                    for m in batch:
                        memory_states.append(m[0])
                        memory_actions.append(m[1])
                        memory_rewards.append(m[2])
                        memory_next_states.append(m[3])
                        memory_dones.append(m[4])

                    nextQs = session.run(self.myNN.output, feed_dict = {self.myNN.input: memory_next_states})
                    for i in range(0, self.batch_size):
                        if batch[i][4]:
                            target_Qs_batch.append(batch[i][2])
                        else:
                            target_Qs_batch.append(batch[i][2] + self.gamma * np.max(nextQs[i]))
                    target_Qs_batch = np.array([each for each in target_Qs_batch])

                    loss, _ = session.run(
                                [self.myNN.loss, self.myNN.optimizer],
                                feed_dict={self.myNN.input: memory_states,
                                           self.myNN.target_Q: target_Qs_batch,
                                           self.myNN.actions: memory_actions})

                    next_state, reward, done, _ = self.env.step(action)
                    total_rewards += reward
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    
                    if (render):
                        self.env.render()
                    
                    current_action = action
                    action = np.zeros(512, dtype=np.int)
                    action[choice] = 1
                    self.memory.add((state, action, reward, next_state, done))
                    if done:
                        next_state = np.zeros((38, 42), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        self.memory.add((state, action, reward, next_state, done))
                        break

                    self.decay_step += 1
                    step += 1
                    state = next_state
                    b = datetime.now()
                    os.system('clear')
                    print("episode: ")
                    print(episode)
                    print("step: ")
                    print(step)
                    print("action: ")
                    print(current_action)
                    print("total_rewards: ")
                    print(total_rewards)
                    print("loss: ")
                    print(loss)
                    print("decay_step: ")
                    print(self.decay_step)
                    print("explore_probability: ")
                    print(explore_probability)
                    print("step time (seconds): ")
                    print((b-a).total_seconds())

                if episode % 5 == 0:
                    save_path = saver.save(session, "./models/model.ckpt")
                    print("Model Saved")


    def play(self, model_path = None):
        with tf.Session() as sess:
            total_test_rewards = []
            
            saver = tf.train.Saver()
            # Load the model
            if (model_path == None):
                saver.restore(sess, "./models/model.ckpt")
            else:
                saver.restore(sess, model_path)

            for episode in range(1):
                total_rewards = 0
                
                state = self.env.reset()
                stacked_frames = deque([np.zeros((38, 42), dtype=np.int) for i in range(4)], maxlen=4)
                state, stacked_frames = stack_frames(stacked_frames, state, True)
                
                print("****************************************************")
                print("EPISODE ", episode)
                
                while True:
                    state = state.reshape((1, *self.state_size))
                    Qs = sess.run(self.myNN.output, feed_dict = {self.myNN.input: state})
                    
                    choice = np.argmax(Qs)
                    action = self.possible_actions[choice]
                    
                    next_state, reward, done, _ = self.env.step(action)
                    self.env.render()
                    
                    total_rewards += reward

                    if done:
                        print ("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        break
                        
                        
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    state = next_state
                    
            self.env.close()


