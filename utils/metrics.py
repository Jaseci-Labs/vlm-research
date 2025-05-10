"""
Evaluation metrics for vision-language models.
"""
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from typing import List, Dict, Union, Any

# Download required NLTK resources
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def calculate_bleu(references: List[str], hypothesis: str) -> Dict[str, float]:
    """
    Calculate BLEU-1,2,3,4 scores
    
    Args:
        references: List of reference texts
        hypothesis: Generated text
        
    Returns:
        Dictionary with BLEU scores
    """
    smoothing = SmoothingFunction().method1
    
    # Tokenize
    tokenized_references = [nltk.word_tokenize(ref.lower()) for ref in references]
    tokenized_hypothesis = nltk.word_tokenize(hypothesis.lower())
    
    # Calculate scores for different n-grams
    bleu_scores = {}
    for i in range(1, 5):
        weights = tuple([1.0/i] * i + [0.0] * (4-i))
        score = sentence_bleu(
            tokenized_references, 
            tokenized_hypothesis,
            weights=weights,
            smoothing_function=smoothing
        )
        bleu_scores[f'bleu-{i}'] = score
    
    return bleu_scores

def calculate_rouge(references: List[str], hypothesis: str) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores
    
    Args:
        references: List of reference texts
        hypothesis: Generated text
        
    Returns:
        Dictionary with ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores against all references and take the best
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0
    }
    
    for reference in references:
        results = scorer.score(reference, hypothesis)
        
        # Update if this reference gives better scores
        if results['rouge1'].fmeasure > scores['rouge-1']:
            scores['rouge-1'] = results['rouge1'].fmeasure
        
        if results['rouge2'].fmeasure > scores['rouge-2']:
            scores['rouge-2'] = results['rouge2'].fmeasure
        
        if results['rougeL'].fmeasure > scores['rouge-l']:
            scores['rouge-l'] = results['rougeL'].fmeasure
    
    return scores

def calculate_meteor(references: List[str], hypothesis: str) -> float:
    """
    Calculate METEOR score
    
    Args:
        references: List of reference texts
        hypothesis: Generated text
        
    Returns:
        METEOR score
    """
    # Tokenize
    tokenized_references = [nltk.word_tokenize(ref.lower()) for ref in references]
    tokenized_hypothesis = nltk.word_tokenize(hypothesis.lower())
    
    # Calculate METEOR score
    score = meteor_score.meteor_score(tokenized_references, tokenized_hypothesis)
    
    return score

def evaluate_predictions(predictions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate predictions with multiple metrics
    
    Args:
        predictions: List of dictionaries with true_caption and pred_caption keys
        
    Returns:
        Dictionary with evaluation metrics
    """
    bleu_scores = {f'bleu-{i}': 0.0 for i in range(1, 5)}
    rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    meteor_scores = []
    
    for pred in predictions:
        reference = [pred['true_caption']]
        hypothesis = pred['pred_caption']
        
        # Calculate metrics
        bleu = calculate_bleu(reference, hypothesis)
        rouge = calculate_rouge(reference, hypothesis)
        meteor = calculate_meteor(reference, hypothesis)
        
        # Accumulate scores
        for k, v in bleu.items():
            bleu_scores[k] += v
        
        for k, v in rouge.items():
            rouge_scores[k] += v
        
        meteor_scores.append(meteor)
    
    # Average scores
    num_samples = len(predictions)
    for k in bleu_scores:
        bleu_scores[k] /= num_samples
    
    for k in rouge_scores:
        rouge_scores[k] /= num_samples
    
    avg_meteor = sum(meteor_scores) / num_samples
    
    # Combine all metrics
    metrics = {
        **bleu_scores,
        **rouge_scores,
        'meteor': avg_meteor
    }
    
    return metrics