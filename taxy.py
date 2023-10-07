import gym
import numpy
import random
from time import sleep

env = gym.make("Taxi-v3", render_mode='ansi').env
q_table_spaces = numpy.zeros([env.observation_space.n, env.action_space.n])

training_episodes = 25000
episodes = 10

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(training_episodes):
    state = env.reset()[0]
    done = False
    reward = 0
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(q_table_spaces[state])

        next_state, reward, done, _, info = env.step(action) 
        
        old_value = q_table_spaces[state, action]
        next_max = numpy.max(q_table_spaces[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table_spaces[state, action] = new_value

        state = next_state
        
    if i % 1000 == 0: print(f"Счетчик: {i}")

print("Обучение завершено. Визуализируем результаты за 100 эпизодов\n")
total_epochs, total_penalties = 0, 0

for i in range(10):
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = numpy.argmax(q_table_spaces[state])
        state, reward, done, _, info = env.step(action)
        if reward == -10:
            penalties += 1
        epochs += 1    
        print(env.render())
        sleep(0.2)

    total_penalties += penalties
    total_epochs += epochs

print(f"Среднее количество шагов за эпизод: {total_epochs / 10}")
print(f"Средний штраф за эпизод: {total_penalties / 10}")