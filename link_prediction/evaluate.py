import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from load_data import *

def get_score(saved_model, node1, node2):
    try:
        embed_1 = saved_model[int(node1)]
        embed_2 = saved_model[int(node2)]
        result = np.dot(embed_1, embed_2)
        return result
    except Exception as e:
        pass


def metric(model, true_edges, false_edges):
    label_list = list()
    predict_list = list()
    num = len(true_edges)
    for edge in true_edges:
        score = get_score(model, edge[0], edge[1])
        predict_list.append(score)
        label_list.append(1)
    for edge in false_edges:
        score = get_score(model, edge[0], edge[1])
        predict_list.append(score)
        label_list.append(0)
    
    # set threshold
    score_sorted = predict_list[:]
    score_sorted.sort()
    threshold = score_sorted[-num]

    y_pred = np.zeros(len(predict_list), dtype=np.int32)
    for i in range(len(predict_list)):
        if predict_list[i] >= threshold:
            y_pred[i] = 1
    y_true = np.array(label_list)
    y_scores = np.array(predict_list)

    return np.mean(roc_auc_score(y_true, y_scores)), np.mean(f1_score(y_true, y_pred))



def link_predict(test_data_true, test_data_false, args):
    print("loading embedding...")
    embeds = np.load(args.out_dir + "/embedding.npy")
    test_embeds = load_test_embedding(args.out_dir + "/embedding.txt", embeds)
    print('test_embeds', test_embeds.shape)

    print("Running link prediction...") 
    mrr, hit30 = metric(test_embeds, test_data_true, test_data_false) 
    return mrr, hit30