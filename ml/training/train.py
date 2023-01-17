import logging
import os

import datasets
import evaluate
import numpy as np
import torch
import transformers
from PIL import ImageFile
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomHorizontalFlip,RandomResizedCrop,ColorJitter,
                                    Resize, ToTensor)
from transformers import (AutoConfig, AutoFeatureExtractor,
                          AutoModelForImageClassification
                          , Trainer, EarlyStoppingCallback,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "digital-art")
model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "training")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "training_out")

def load_dataset_images():
    train_val_split = 0.20

    dataset = datasets.load_dataset("imagefolder",data_dir=data_dir, split=f'train', task="image-classification",)

    dataset = dataset.rename_column("image", "pixel_values")
    # Shuffle dataset
    dataset = dataset.shuffle()

    # Split dataset and save
    dataset = dataset.train_test_split(train_val_split)

    return (dataset["train"], dataset["test"])

def load_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

    return last_checkpoint
    
def get_transforms(feature_extractor, size):
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(), 
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["pixel_values"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["pixel_values"]]
        return example_batch
    
    return train_transforms, val_transforms

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def train(train_dataset, test_dataset):
    # Set hyperparameters
    training_args = TrainingArguments(
        output_dir="output_dir", 
        evaluation_strategy="steps", 
        do_train=True, 
        do_eval=True, 
        overwrite_output_dir=False, 
        log_level="warning",
        num_train_epochs=12, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end = True,
        eval_steps=5000,
        save_steps=5000,
        metric_for_best_model = "accuracy",
    )

    # Set logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = load_checkpoint(training_args)

    # Set labels
    labels = train_dataset.features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
    config = AutoConfig.from_pretrained(
       model,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        problem_type="single_label_classification",
    )
    model = AutoModelForImageClassification.from_pretrained(
       model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        os.path.join(model, "preprocessor_config.json"),
        ignore_mismatched_sizes=True,
    )
    
    if "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"], feature_extractor.size["width"])

    train_transforms, val_transforms = get_transforms(feature_extractor, size)

    # Set the training transforms
    train_dataset.set_transform(train_transforms)
    test_dataset.set_transform(val_transforms)

        # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

        
if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    d_train, d_test = load_dataset_images()
    train(d_train, d_test)