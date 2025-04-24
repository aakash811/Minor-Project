from rouge_score import rouge_scorer

def evaluate_rouge(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)
