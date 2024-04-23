import warnings

import keras.models

import __init__

import gym
import gym_snowfight  # there may be problems during import. check __init__.py in this directory for more.
from gym_snowfight.wrappers import RelativePosition, Compactify, CompactifyMore
from cmdargs import args
from output import Output

import time

import gym
import pygame

import numpy as np
from cmdargs import args

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D
from keras.optimizers import adam_v2

import copy
import threading
from qlibrary import ExperienceReplay, Experience

size = args.size
mode = args.mode
ws   = args.window_size


save_interval = 100
num_threads = 1 if args.mode == 'human' else args.n_threads  # 10 threads for home pc.
update_q_interval = 12 if args.mode == 'human' else num_threads*2
buffer_capacity = 8192
best_capacity = 0
min_buffer_length = 512
buffer = ExperienceReplay(buffer_capacity, best_capacity=best_capacity)
learning_rate = .0001
lr_decay = 1
# the smaller the step, the less possible it is that it deviates from previous too much
# often, with lr=.001, the model reached and missed the optimum.
# hyperparameter tuning
gamma = .995  # optimal value is .95 after testing.
epsilon_decay = 1-1E-2  #(1-5*1E-6)  # tried different values.
epsilon_min = 1E-7
# playing with hyperparameters.
# goal:
# 1) do well when training from scratch.
# 2) do well when training from loaded model.


warnings.filterwarnings("ignore")
# for random agent, use play v1 script with -m human_rand
assert mode != "human_rand"

window_size = args.window_size
max_steps = args.max_steps

env = gym.make('gym_snowfight/SnowFight', render_mode=mode, size=size, window_size=window_size, max_enemies=args.enemies)
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
env = CompactifyMore(env)   #Compactify(env)
#RelativePosition(env)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps
else:
    env.metadata["render_fps"] = 1000000000

env.action_space.seed(args.seed)

input_file = args.file
file = args.output_file
if file is None:
    file = f"data/qtable_{time.strftime('%Y%m%d%H%M')}"

output = Output(file, 'model', output_every_n=save_interval)  # does any output job.
folder = output.dir

# threading
envs = [copy.deepcopy(env) for i in range(num_threads)]


def end():
    if episode > 0:
        print(f"Average score: {total_score / episode:.2f}\n" +
              f"Maximum score: {max_score:.2f}\n" +
              f"Average steps: {total_steps / episode:.2f} ([{min_steps_achieved} to {max_steps_achieved}])")

    env.close()


# DQN model
num_classes = env.action_space.n
input_shape = env.observation_space.shape

if input_file is not None:
    print(f"model loaded from {input_file}")
    model = tf.keras.models.load_model(input_file, compile=False)
    print(model.summary())
    epsilon = 1E-5
else:
    print("creating a new model.")
    "Dimensionality reduction by obtaining Q-value rows by using a neural network."
    model = Sequential(
        [
            tf.keras.Input(shape=input_shape),
            Reshape((-1,)),
            Dense(128, activation='relu'),   # 128
            Dense(128, activation='relu'),   # 128
            Dense(num_classes)
        ]
    )
    model.build()
    epsilon = .5

target = keras.models.clone_model(model)

opt = adam_v2.Adam(learning_rate=learning_rate)


print(model.summary())
with open(f"{folder}/model_structure.txt", "a+") as f:
    f.write(f"Model trained from loaded file: {input_file}\n")
    f.write(f"Parameters: \tbuffer size: {buffer_capacity}\n")
    f.write(f"\t\t\t\t* best capacity: {best_capacity}\n")
    f.write(f"\t\t\t\t* gamma: {gamma}, epsilon: {epsilon} (decay = {epsilon_decay}, min = {epsilon_min})\n")
    f.write(f"\t\t\t\t* lr: {learning_rate} (decay = {lr_decay})\n")
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Test the agent
test_episodes = args.episodes
max_steps = args.max_steps

episode = 0
total_score = 0
max_score = 0
total_steps = 0
min_steps_achieved = (2 << 15)
max_steps_achieved = 0

running = True
step = 0

high = -4

test_rwd = 0

# threading
class PlayModel (threading.Thread):

    def __init__(self, env, episode, inp_model=model, eps_greedy=True, output=False):
        threading.Thread.__init__(self)
        self.env = env
        self.episode = episode
        self.model = inp_model
        self.eps_greedy = eps_greedy
        self.output = output

    @staticmethod
    def vectorize(inp: np.ndarray):
        return np.expand_dims(inp, axis=0)

    def run(self):
        global max_score, total_score, epsilon, \
            max_steps_achieved, min_steps_achieved, total_steps, output, high, test_rwd
        env = self.env
        episode = self.episode
        state = env.reset(seed=args.seed)[0]  # [0] for observation only
        total_testing_rewards = 0

        for step in range(max_steps):
            if args.mode != 'rgb_array':
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        output.output_model(self.model)
                        end()
                        exit()

            "Obtain Q-values from network."
            q_values = self.model(self.vectorize(state))

            "Select action using epsilon-greedy strategy."
            sample_epsilon = np.random.rand()
            if sample_epsilon <= epsilon and self.eps_greedy:
                action = env.action_space.sample()
            else:
                qv = q_values[0].numpy()
                mx = np.amax(qv)
                space = np.array(np.nonzero(qv == mx)).reshape((-1,))
                if len(space) == 0:
                    action = env.action_space.sample()
                else:
                    action = np.random.choice(space, size=1)[0]
            "Obtain q-value for the selected action."
            q_value = q_values[0, action]
            # print(q_values)

            "Deterimine next state."
            new_state, reward, done, truncated, info = env.step(action)  # take action and get reward
            buffer.append(Experience(state, action, reward, done, new_state))
            state = new_state

            if step % 1000 == 999:
                print(f"step {step} for episode {self.episode}")

            if info['score'] > high:
                epsilon *= epsilon_decay
                if epsilon < epsilon_min:
                    epsilon = epsilon_min

                high = info['score']

            # print(state, action)
            if done or truncated:
                test_rwd += info['score']
                total_score += info['score']
                max_score = max(max_score, info['score'])

                if self.output:
                    output.log(done, episode, step, info, model=model, do_output=False, epsilon=epsilon)
                else:
                    break

                total_steps += step
                max_steps_achieved = max(max_steps_achieved, step)
                min_steps_achieved = min(min_steps_achieved, step)
                break

    def join(self):
        threading.Thread.join(self)


class TrainModel (threading.Thread):

    def __init__(self, experience, episode):
        threading.Thread.__init__(self)
        self.experience = experience
        self.episode = episode

    @staticmethod
    def vectorize(inp: np.ndarray):
        return np.expand_dims(inp, axis=0)

    def run(self):
        global max_score, total_score, epsilon, \
            max_steps_achieved, min_steps_achieved, total_steps, output, high
        experience = self.experience
        episode = self.episode
        state, action, reward, done, new_state = experience

        with tf.GradientTape() as tape:  # tracing and computing the gradients ourselves.
            "Obtain Q-values from network."
            q_values = model(state)

            "Obtain q-value for the selected action."
            q_value = tf.gather(q_values, tf.constant(action), axis=1)#q_values[action]
            #print(q_values)

            "From the Q-learning update formula, we have:"
            "   Q'(S, A) = Q(S, A) + a * {R + λ argmax[a, Q(S', a)] - Q(S, A)}"
            "Target of Q' is given by: "
            "   R + λ argmax[a, Q(S', a)]"
            "Hence, MSE loss function is given by: "
            "   L(w) = E[(R + λ argmax[a, Q(S', a, w)] - Q(S, a, w))**2]"
            next_q_values = tf.stop_gradient(target(new_state))
            next_actions = tf.math.argmax(next_q_values, 1)
            next_q_value = tf.gather(next_q_values, next_actions, axis=1)

            observed_q_value = reward + (gamma * next_q_value)
            loss = (observed_q_value - q_value) ** 2

            "Computing and applying gradients"
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    def join(self):
        threading.Thread.join(self)


def nothing(*aargs):
    pass


def repeat(thread_func, before=nothing, after=nothing, afterloop=nothing, n_threads=num_threads, n=1):
    """ :before: a function that is done before repeating.
        :thread_func: a function that outputs a thread.
        :after: a function that is done after repeating
        :n_threads: number of threads. default to be num_threads.
        :n: how many iterations to loop for. default 1."""
    train_threads = []
    before()
    for j in range(n):
        for i in range(n_threads):
            thread = thread_func(i)  # outputs a thread
            thread.start()
            train_threads.append(thread)
            afterloop()
        [trainThread.join() for trainThread in train_threads]
        train_threads = []
    after(n_threads*n)

print("Training started ...")
trainThreads = []
updated = 0
episode = 0
train_episode = 0
best_rwd = -1000
this_rwd = 0
bbest_rwd = -1000
best_model = keras.models.clone_model(model)

def reset_test_rwd():
    global test_rwd
    test_rwd = 0
def collect_data_finish(n=num_threads):
    global this_rwd, episode
    this_rwd = test_rwd / n
def evaluate_best_finish(n=num_threads):
    global bbest_rwd
    bbest_rwd = test_rwd / n
def afterloop():
    global episode
    episode += 1


while episode < test_episodes:
    repeat(
        lambda i: PlayModel(envs[i], episode, inp_model=model, eps_greedy=True,
                            output=True),
        before=reset_test_rwd,
        after=collect_data_finish,
        n_threads=num_threads,
        afterloop=afterloop,
        n=1
    )

    if len(buffer) > min_buffer_length:
        if train_episode / update_q_interval > updated:
            # target.set_weights(model.get_weights())
            # stores playing statistics of the best model until now.
            # models may deteriorate, so we must do this to obtain good results.
            if best_rwd < this_rwd:
                bbest_rwd = 0
                repeat(
                    lambda i: PlayModel(envs[i], episode, inp_model=model, eps_greedy=False),
                    before=reset_test_rwd,
                    after=evaluate_best_finish,
                    n_threads=num_threads,
                    n=2  # current best model is trained with n = 2.
                )
                # this step is to get the best score, while checking if the
                # model is really the best one
                # or performs a little bit better due to randomness.

                if bbest_rwd >= best_rwd:
                    output.output_model(-episode, best_model)
                    best_model.set_weights(model.get_weights())
                    best_rwd = bbest_rwd
                    print(f"Model updated. best rwd: {best_rwd}")
                    output.output_model(-1, best_model)
                    output.concat({'episode': [episode], 'best': [best_rwd], 'good': [bbest_rwd]})
                    output.output_img(episode=-1)
                else:
                    print(f"Model not updated. best rwd: {best_rwd}")
                    output.concat({'episode': [episode], 'best': [best_rwd], 'good': [bbest_rwd]})
                    output.output_img(episode=-1)
            # target, model = model, target
            target = model
        if best_rwd - this_rwd > max(1, best_rwd/2):
            print(f"bad model (score={this_rwd}). "
                  f"feed with experiences of the good model: {best_rwd}."
                  f"repeated {min(3, int(best_rwd-this_rwd))} times")
            repeat(
                lambda i: PlayModel(envs[i], episode, inp_model=best_model, eps_greedy=True),
                before=reset_test_rwd,
                after=lambda i: None,
                n_threads=num_threads,
                n=min(3, int(best_rwd-this_rwd))
            )
        for j in range(num_threads):
            thread = TrainModel(buffer.sample(512), train_episode)
            thread.start()
            trainThreads.append(thread)
            if j == num_threads - 1:
                [trainThread.join() for trainThread in trainThreads]
                trainThreads = []
            if train_episode % save_interval > save_interval - 5:
                output.concat({'episode': [episode], 'best': [np.NAN], 'good': [np.NAN]})
                output.output_img(episode=-1)
            train_episode += 1



end()
