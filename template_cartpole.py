import gym
import tensorflow as tf
import numpy as np

problem = 'CartPole-v1'
env = gym.make(problem)

nS = env.observation_space.shape[0]
nA = env.action_space.n
action = env.action_space.sample()


def get_model():
	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(128,activation='relu')(inputs)
	out = tf.keras.layers.Dense(128,activation='relu')(out)
	outputs = tf.keras.layers.Dense(1)(out)
	model = tf.keras.Model(inputs,outputs)
	return model

def policy(state):
	
	p_action = actor_model(state)
	act_prob = np.squeeze( p_action.numpy() )
	action = np.random.choice(nA,p=act_prob )
	return action

model = get_model()

optimizer = tf.keras.optimizers.Adam(  )

total_eps = 200

for ep in range(total_eps):

	prev_state = env.reset()

	while True:

		env.render()

		q = rewa

		action = policy( tf_prev_state )
		state, reward, done, info = env.step(action)


		if done:
			print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_r))
			break
		else:
			prev_state = state


env.close()



