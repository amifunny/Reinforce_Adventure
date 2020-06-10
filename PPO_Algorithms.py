import gym
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Discrete action environment
problem = 'CartPole-v1'
env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Define a policy approximator
# Takes State as input and give actions
def get_policy_model():
  init = tf.keras.initializers.GlorotNormal()
  inputs = tf.keras.layers.Input( shape=(num_states) )
  out = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=init)(inputs)
  out = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=init)(out)
  outputs = tf.keras.layers.Dense(num_actions,activation='softmax',kernel_initializer=init)(out)
  model = tf.keras.Model(inputs,outputs)
  return model


def get_value_model():
  init = tf.keras.initializers.GlorotNormal()
  inputs = tf.keras.layers.Input( shape=(num_states) )
  out = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=init)(inputs)
  out = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=init)(out)
  outputs = tf.keras.layers.Dense(1,kernel_initializer=init)(out)
  model = tf.keras.Model(inputs,outputs)
  return model


def policy(state):
  
  p_action = tf.squeeze( policy_model(state) )
  act_prob = np.squeeze( p_action.numpy() )

  # Choose action using `act_prob`
  action = np.random.choice(num_actions,p=act_prob )

  old_prob = ( p_action[action] )

  return action,old_prob


policy_model = get_policy_model()
value_model = get_value_model()

value_optimizer = tf.keras.optimizers.Adam( 0.001 )
policy_optimizer = tf.keras.optimizers.Adam( 0.001 )

# Num of episodes to sample
total_episodes = 2000
# Used to decide cliping
epsilon = 0.2
# Size of random sampled batch
default_batch_size = 16
# Num of epochs to train
epochs = 10
# Store reward history
ep_reward_list = []
avg_reward = []
# Interval of episodes for training
ep_to_wait = 10

for episode in range(total_episodes):

  prev_state = env.reset()
  episodic_r = 0

  # Check if interval is over
  if episode%ep_to_wait==0:
    state_list = []
    action_list = []
    reward_list = []
    next_state_list = []
    old_prob_list = []

  while True:

    # env.render()

    tf_prev_state = tf.expand_dims( tf.convert_to_tensor(prev_state) , 0 )

    action,old_prob = policy( tf_prev_state )
    state, reward, done, info = env.step(action)

    tf_state = tf.expand_dims( tf.convert_to_tensor(state) , 0 )

    episodic_r += reward

    state_list.append( prev_state )
    action_list.append( action )
    next_state_list.append( state )
    old_prob_list.append( old_prob )

    if done:
      print("Episode * {} * ==> Total Reward is ==> {}".format(episode,episodic_r))

      reward = -1
      reward_list.append( reward )
            
      break
    else:
      prev_state = state

    reward_list.append( reward )

  
  if episode%ep_to_wait==0 and episode!=0:

    states = tf.convert_to_tensor( state_list )
    actions = tf.expand_dims( tf.convert_to_tensor( action_list ) , -1 )
    rewards = tf.expand_dims( tf.convert_to_tensor( reward_list ) , -1 )
    rewards = tf.cast( rewards , tf.float32 )
    next_states = tf.convert_to_tensor( next_state_list )
    old_probs = tf.expand_dims( tf.convert_to_tensor( old_prob_list ) , -1 )

    experience_size = states.shape[0]
    batch_size = max(experience_size//5,default_batch_size)
    for i in range( epochs ):

      sampled_indices = tf.convert_to_tensor( np.random.choice( experience_size , batch_size ) )

      l_state = tf.gather( states , sampled_indices )
      l_action  = tf.gather( actions , sampled_indices )
      l_reward  = tf.gather( rewards , sampled_indices )
      l_next_state  = tf.gather( next_states , sampled_indices )
      l_old_prob = tf.gather( old_probs , sampled_indices )

      with tf.GradientTape() as tape:

        advantage = l_reward + 0.99 * value_model( l_next_state ) - value_model( l_state )
        value_loss = tf.math.reduce_mean( 0.5*tf.square( advantage ) )
        print("Critic Loss --> {}".format(value_loss))

      value_grad = tape.gradient( value_loss , value_model.trainable_variables )  
      value_optimizer.apply_gradients( zip( value_grad , value_model.trainable_variables ) )

      with tf.GradientTape() as tape:

        prob_dist = tf.squeeze( policy_model(l_state) )
        new_prob = ( tf.expand_dims( tf.gather_nd( prob_dist , l_action , batch_dims=1 ) , -1 ) )

        rt = new_prob/l_old_prob

        first_term = rt*advantage
        second_term = tf.clip_by_value( rt , 1-epsilon , 1+epsilon )*advantage
        policy_loss_t = tf.math.minimum( first_term , second_term )
        policy_loss = -1.0*tf.math.reduce_mean( policy_loss_t )

        print("Policy Loss --> {}".format(policy_loss))

      policy_grad = tape.gradient( policy_loss , policy_model.trainable_variables ) 
      policy_optimizer.apply_gradients( zip( policy_grad , policy_model.trainable_variables ) )

    
  ep_reward_list.append( episodic_r )
  avg_reward.append( np.mean( ep_reward_list[-40:] ) )
  


# # Save with pickle
ql_results = open('results','wb')
pickle.dump( ep_reward_list , ql_results )                      

# Plot a Graph
# Episodes vs Rewards
# Now that what you call a beautiful graph
# You may notice Actor-Critic perform better than simple Q-Learning and Vanilla Policy Gradient
plt.plot( avg_reward )
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward') 
plt.show()

env.close()