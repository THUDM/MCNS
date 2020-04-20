import numpy as np
import time
import tensorflow as tf
from collections import defaultdict
from scipy.stats import norm


def get_length(walks):
    length = 0
    for key in walks.keys():
        length += len(walks[key])
    return length

def negative_sampling(model, sess, candidates, start_given, q_1_dict, N_steps, N_negs, node1, node2, args):
    distribution = [0.01] * 100
    # distribution = norm.pdf(np.arange(0,100,1), 50, 10)
    # distribution = norm.pdf(np.arange(0,100,1), 0, 50)
    # distribution = norm.pdf(np.arange(0,100,1), 100, 100)
    # distribution = norm.pdf(np.arange(0,100,1), 50, 100)
    distribution = [i/np.sum(distribution) for i in distribution]
    
    if start_given is None:
        start = np.random.choice(list(q_1_dict.keys()),args.batch_size)  # random init (user and item)
    else:
        start = start_given
    count = 0
    cur_state = start
    user_list = node1
    walks = defaultdict(list)
    generate_examples = list()
    while True:
        y_list = list()
        q_probs_list = list()
        q_probs_next_list = list()
        t0 = time.time()
        count += 1
        sample_num = np.random.random()
        if sample_num < 0.5:
            y_list = np.random.choice(list(q_1_dict.keys()), len(cur_state), p=list(q_1_dict.values()))
            q_probs_list = [q_1_dict[i] for i in y_list]
            q_probs_next_list = [q_1_dict[i] for i in cur_state]
        else:
            tt_0 = time.time() 
            for i in cur_state:
                y = np.random.choice(candidates[i], 1, p=distribution)[0]
                y_list.append(y)
                index = candidates[i].index(y)
                q_probs = distribution[index]
                q_probs_list.append(q_probs)
                node_list_next = candidates[y]
                if i in node_list_next:
                    index_next = node_list_next.index(i)
                    q_probs_next = distribution[index_next]
                else:
                    q_probs_next = q_1_dict[i]
                q_probs_next_list.append(q_probs_next) 

            # print("time_1", time.time() - tt_0) 
        u = np.random.rand()
        tt = time.time() 
        p_probs = sess.run(model.p_probs, feed_dict={model.inputs1:user_list, model.inputs2:y_list , model.batch_size: len(user_list), model.number: len(user_list)})
        p_probs_next = sess.run(model.p_probs, feed_dict={model.inputs1:user_list, model.inputs2:cur_state , model.batch_size: len(user_list), model.number: len(user_list)})

        A_a_list = np.multiply(np.array(p_probs), np.array(q_probs_next_list))/ np.multiply(np.array(p_probs_next), np.array(q_probs_list))
        next_state = list()
        next_user = list()
        if count > N_steps:
            for i in list(range(len(cur_state))):
                if y_list[i] >= args.user_num:
                    walks[user_list[i]].append(y_list[i])
                else:
                    next_state.append(y_list[i])
                    next_user.append(user_list[i])
            cur_state = next_state
            user_list = next_user
        else:
            for i in list(range(len(cur_state))):
                A_a = A_a_list[i]                        
                alpha = min(1, A_a)
                if u < alpha:
                    next_state.append(y_list[i])
                else:
                    next_state.append(cur_state[i])
            cur_state = next_state
        # print("time_2", time.time() - tt) 
        length = get_length(walks)  

        if length == args.batch_size:
            # print("count", count)
            generate_examples = list()
            for user in node1:
                d = walks[user]
                if len(d) == 1:
                    generate_examples.append(d[0])    
                else:
                    generate_examples.append(d[0])
                    del walks[user][0]
            break
        else:
            continue  
    return generate_examples
    
