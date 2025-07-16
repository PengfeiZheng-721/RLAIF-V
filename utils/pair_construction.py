import numpy as np
from itertools import combinations
from collections import defaultdict


# --- FIX 1: Correctly calculate scores from a list ---
def func_yes_prob(item_scores):
    # The input 'item_scores' is a list of floats, so we average them.
    if not item_scores:
        return 0.0
    return sum(item_scores) / len(item_scores)

def func_no_prob(item_scores):
    # The "no" probability is simply 1 minus the average "yes" probability.
    if not item_scores:
        return 1.0
    avg_yes_prob = sum(item_scores) / len(item_scores)
    return 1.0 - avg_yes_prob


def get_pred_scores(pred_data_addscores, func):
    pred_scores = []
    for item in pred_data_addscores:
        pred_scores.append(func(item['scores']))
    return pred_scores


# --- FIX 2: Correctly group answers using a reliable key ---
def get_dsid_to_question_id(pred):
    # Group different generated answers by their shared "raw_question" text.
    dsid_to_question_ids = defaultdict(list)
    for item in pred:
        # The raw_question text is the reliable key for grouping.
        key = item.get('raw_question')
        if key is None:
            continue
        
        # We collect the unique IDs of each generated answer for that question.
        dsid_to_question_ids[key].append(item['question_id'])

    dsid_to_question_ids = {key: list(set(value)) for key, value in dsid_to_question_ids.items()}
    return dsid_to_question_ids


def pair_data_judge(data_item_0, data_item_1, diff):
    score_diff = data_item_0['score'] - data_item_1['score']
    if abs(score_diff) >= diff:
        if score_diff < 0:
            chosen = data_item_1
            rejected = data_item_0
        else:
            chosen = data_item_0
            rejected = data_item_1
        return {'chosen': chosen, 'rejected': rejected}
    else:
        return None

def get_pair_data(quesid_to_scores, dsid_to_question_ids, diff):
    pair_data = []

    for key, question_ids in dsid_to_question_ids.items():

        potential_pairs = []

        # This combination logic will now work because the grouping is correct.
        for comp_idx1, comp_idx2 in combinations(question_ids, 2):

            ans_1_score = quesid_to_scores.get(comp_idx1, 0)
            ans_2_score = quesid_to_scores.get(comp_idx2, 0)

            ans_1 = {"question_id": comp_idx1, "score": ans_1_score}
            ans_2 = {"question_id": comp_idx2, "score": ans_2_score}

            pair = pair_data_judge(ans_1, ans_2, diff=diff)

            if pair is not None:
                potential_pairs.append(pair)


        for chosen_pair in potential_pairs:
            chosen_pair_data = {
                # The "key" is now the raw_question text.
                "ds_question_id": key, 
                "chosen": chosen_pair['chosen'],
                "rejected": chosen_pair['rejected'],
            }

            pair_data.append(chosen_pair_data)

    return pair_data


def get_pairs_inner(pred_data_addscores, diff=1, return_infos=False):
    def pred_scores_to_class(pred):
        pred_scores_yes = np.array(get_pred_scores(pred_data_addscores, func=func_yes_prob))
        pred_scores_no = np.array(get_pred_scores(pred_data_addscores, func=func_no_prob))

        pred_cls = pred_scores_yes > pred_scores_no

        pred_addcls = []
        for i,item in enumerate(pred):
            item['pred_label'] = int(pred_cls[i])
            pred_addcls.append(item)

        return pred_addcls

    def get_pred_ans_scores(pred_addcls):
        pred_quesid_to_yesprob_list = defaultdict(list)
        pred_quesid_to_judge = defaultdict(dict)
        for item in pred_addcls:
            question_id = item['question_id']
            yes_prob = func_yes_prob(item['scores'])
            pred_quesid_to_yesprob_list[question_id].append(yes_prob)
            raw_question = item['raw_question'] if 'raw_question' in item else item['question']
            pred_quesid_to_judge[question_id][raw_question] = '1' if item['pred_label'] == True else '0'

        pred_quesid_to_scores = {key: sum(value) / len(value) for key, value in pred_quesid_to_yesprob_list.items()}

        return pred_quesid_to_scores, pred_quesid_to_judge

    pred_addcls = pred_scores_to_class(pred_data_addscores)
    pred_quesid_to_scores, pred_quesid_to_judge = get_pred_ans_scores(pred_addcls)

    dsid_to_question_ids = get_dsid_to_question_id(pred_data_addscores)
    
    # --- DEBUGGING PRINT ADDED HERE TEMPORARILY ---
    print("\n--- DEBUG: Checking Answer Groups ---")
    print(f"Total number of unique questions found for pairing: {len(dsid_to_question_ids)}")
    for i, (key, value) in enumerate(dsid_to_question_ids.items()):
        if i >= 5: break
        print(f"Group {i+1}: Question='{key[:70]}...', Answer_IDs={value}")
    print("--- END DEBUG ---\n")
    # --- END DEBUGGING PRINT ---

    pair_data = get_pair_data(pred_quesid_to_scores, dsid_to_question_ids, diff)

    if return_infos:
        return pair_data, pred_quesid_to_judge, pred_addcls
    else:
        return pair_data