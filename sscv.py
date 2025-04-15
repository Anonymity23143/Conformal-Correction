import pandas as pd
import numpy as np

# def sort_sum(scores):
#     I = scores.argsort(axis=1)[:,::-1]
#     ordered = np.sort(scores,axis=1)[:,::-1]
#     cumsum = np.cumsum(ordered,axis=1) 
#     return I, ordered, cumsum

# strata = [[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]


def get_violation(prediction_sets, label, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    # for logit, target in loader_paramtune:
    # compute output
    # output, S = cmodel(logit) # This is a 'dummy model' which takes logits, for efficiency.
    # measure accuracy and record loss
    size = np.array([x.size for x in prediction_sets])
    # I, _, _ = sort_sum(logit.numpy()) 
    correct = np.zeros_like(size)
    for j in range(correct.shape[0]):
        correct[j] = int( label[j] in list(prediction_sets[j]) )
    batch_df = pd.DataFrame({'size': size, 'correct': correct})
    # df = df.append(batch_df, ignore_index=True)
    df = pd.concat([df, batch_df], ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[ (df['size'] >= stratum[0]) & (df['size'] <= stratum[1]) ]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation # the violation