import sys, math
import numpy as np
import time

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

#Created by https://github.com/shivaverma

import gym
from gym import spaces
from gym.utils import seeding
import skvideo.io
from keras.models import Sequential
from keras.layers import Dense
from lunar_lander import *

# Initialize training data
X_train, y_train = [],[]
frames = []

if __name__=="__main__":
    # Initialize lunar lander environment
    env = LunarLanderContinuous()
    prev_s = env.reset()
    total_reward = 0
    steps = 0
    a = np.array([0.0,0.0])
    modelTrained = False
    model = nnmodel(10  )
    print(model.summary())
    tr = 0
    prev_r = 0
    train_rewards= []
    training_thr = 5000
    total_itrs = 50000
#    training_thr = 200
#    total_itrs = 1000
    successful_steps = []

    train_start=time.time()
    while steps <= total_itrs:
        new_s, r, done, info = env.step(a)
        X_train.append(list(prev_s)+list(a))
        y_train.append(r)
        train_rewards.append((steps,r))

        # This 'if' statement determines how often your model is re-trained

        if steps >= training_thr and steps %1000 ==0:
            # re-train a model
            print("training model model")
            modelTrained = True
            model.fit(np.array(X_train),np.array(y_train).reshape(len(y_train),1), epochs = 20, batch_size=64)

        if modelTrained:
            maxr = -1000
            maxa = None
            for i in range(100):
                a1 = np.random.randint(-1000,1000)/1000
                a2 = np.random.randint(-1000,1000)/1000
                testa = [a1,a2]
                r_pred = model.predict(np.array(list(new_s)+list(testa)).reshape(1,len(list(new_s)+list(testa))))
                if r_pred > maxr:
                    maxr = r_pred
                    maxa = testa
            a = np.array(maxa)

        else:
            a = np.array([np.random.randint(-1000,1000)/1000,\
                 np.random.randint(-1000,1000)/1000])

        if steps %100 == 0:
            print("At step ", steps)
            print("reward: ", r)
            print("total rewards ", tr)
        prev_s = new_s
        prev_r = r

        if (steps >= training_thr and tr < -500) or done:
            print(prev_s)
            if done and prev_r >= 200:
                successful_steps.append(steps)
                print("Successful Landing!!! ",steps)
                print("Total successes are: ", len(successful_steps))
            env.reset()
            tr = 0

        tr = tr + r

        steps += 1
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        if steps >= training_thr and steps %1000 == 0:
            fname = "./outputs/videos/frame"+str(steps)+".mp4"
            skvideo.io.vwrite(fname, np.array(frames))
            del frames
            frames = []
            
    train_end =time.time()
    training_time = train_end-train_start
    np.savetxt('./outputs/training_rewards.txt',train_rewards)
        
if __name__=="__main__":
    env = LunarLanderContinuous()
    prev_s = env.reset()
    total_reward = 0
    steps = 0
    a = np.array([0.0,0.0])
    modelTrained = True
    tr = 0
    prev_r = 0
    pred_rewards = []
    total_itrs = 5000
    
    pred_start = time.time()
    while steps <= total_itrs:

        new_s, r, done, info = env.step(a)

        if modelTrained:
            maxr = -1000
            maxa = None
            for i in range(100):
                a1 = np.random.randint(-1000,1000)/1000
                a2 = np.random.randint(-1000,1000)/1000
                testa = [a1,a2]
                r_pred = model.predict(np.array(list(new_s)+list(testa)).reshape(1,len(list(new_s)+list(testa))))
                if r_pred > maxr:
                    maxr = r_pred
                    maxa = testa
            pred_rewards.append((steps,maxr))
            a = np.array(maxa)

        else:
            a = np.array([np.random.randint(-1000,1000)/1000,\
                 np.random.randint(-1000,1000)/1000])

        if steps %100 == 0:
            print("At step ", steps)
            print("reward: ", r)
            print("total rewards ", tr)
        prev_s = new_s
        prev_r = r
        if done:
            print(prev_s)
            if done and prev_r >= 200:
                print("Successful Landing!!! ",steps)
                print("Total successes are: ", len(successful_steps))
            env.reset()
            tr = 0
        tr = tr + r

        steps += 1
        env.render(mode='human')

    pred_end = time.time()
    prediction_time=pred_end-pred_start
    np.savetxt("./outputs/pred_rewards.txt",pred_rewards)
    
    print("--I--Training Time=%.2f seconds" % training_time)
    print("--I--Prediction Time=%.2f seconds" % prediction_time)
