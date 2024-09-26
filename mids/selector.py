import torch

def make_decision(answers_result, scores):
    """
    Select the best answer with highest match score
    """
    match_scores = []
    preds = []
    for i in range(len(answers_result)):
        if answers_result[i] == 'real':
            match_scores.append(scores[i][0].item()/(scores[i][0].item() + scores[i][2].item()))
            preds.append(0)
        else:
            match_scores.append(scores[i][3].item()/(scores[i][3].item() + scores[i][1].item()))
            preds.append(1)

    scores = torch.tensor(match_scores, device=scores.device)
    best_answer_idx = torch.argmax(scores).item()
    pred = preds[best_answer_idx]
    match_score = match_scores[best_answer_idx]
    forgery_score = match_score if pred == 1 else 1-match_score
    return best_answer_idx, pred, match_score, forgery_score


def make_decision_batch(answers_result, scores, chunk_size=3):
    answers_result = [answers_result[i:i + chunk_size] for i in range(0, len(answers_result), chunk_size)]

    best_answer_idxs = []
    preds = []
    match_scores = []
    forgery_scores = []

    for b in range(len(answers_result)):
        best_answer_idx, pred, match_score, forgery_score = make_decision(answers_result[b], scores[b])
        best_answer_idxs.append(best_answer_idx)
        preds.append(pred)
        match_scores.append(match_score)
        forgery_scores.append(forgery_score)
    return best_answer_idxs, preds, match_scores, forgery_scores