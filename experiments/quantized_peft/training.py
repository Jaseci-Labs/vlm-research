def evaluate_model(cfg: DictConfig, model, tokenizer, test_dataset):
    """
    Evaluate the model on test data
    
    Args:
        cfg: Hydra configuration
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        
    Returns:
        Dictionary of evaluation results
    """
    # Switch model to inference mode
    FastVisionModel.for_inference(model)
    
    # Process test dataset
    results = []
    
    for i, sample in enumerate(test_dataset):
        image = sample["image"]
        true_caption = sample["caption"]
        
        # Generate prediction
        pred_caption = generate_prediction(cfg, model, tokenizer, image)
        results.append({
            "image_id": i,
            "true_caption": true_caption,
            "pred_caption": pred_caption
        })
        
        # Log sample images with predictions
        if i < 10 and cfg.wandb.mode != "disabled":
            log_image_with_prediction(image, true_caption, pred_caption, i)
    
    # Import metrics calculation functions
    from core.metrics import evaluate_predictions
    
    # Calculate standard NLG metrics
    nlg_metrics = evaluate_predictions(results)
    
    # Compute additional metrics
    avg_len = sum(len(r["pred_caption"].split()) for r in results) / len(results)
    
    metrics = {
        "eval/num_samples": len(results),
        "eval/avg_response_length": avg_len
    }
    
    # Add NLG metrics to our metrics dict
    for key, value in nlg_metrics.items():
        metrics[f"eval/{key}"] = value
    
    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        
        # Create a wandb table with results
        columns = ["image_id", "true_caption", "pred_caption"]
        data = [[r["image_id"], r["true_caption"], r["pred_caption"]] for r in results]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"eval/predictions": table})
    
    return metrics