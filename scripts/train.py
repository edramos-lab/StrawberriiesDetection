import sys, os
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import json, cv2, random
from torch.utils.tensorboard import SummaryWriter
import shutil

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_setup, launch, HookBase
from detectron2.config import get_cfg
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import argparse


class TensorboardLoggerHook(HookBase):
    def __init__(self, trainer, writer):
        self.trainer = trainer
        self.writer = writer

    def after_step(self):
        storage = self.trainer.storage.latest()
        metrics = {key: storage[key] for key in storage.keys() if key in storage}
        for key, value in metrics.items():
            if isinstance(value, (tuple, list)):
                value = value[0]  
            self.writer.add_scalar(f"train/{key}", value, self.trainer.iter)


class ValidationHook(HookBase):
    def __init__(self, trainer, writer, eval_period):
        self.trainer = trainer
        self.writer = writer
        self.eval_period = eval_period

    def after_step(self):
        if self.trainer.iter % self.eval_period == 0 and self.trainer.iter > 0:
            evaluator = COCOEvaluator("test", self.trainer.cfg, False, output_dir=self.trainer.cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.trainer.cfg, "test")
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            self.writer.add_scalar(f"validation/{key}/{sub_key}", sub_value, self.trainer.iter)
                elif isinstance(value, (int, float)):
                    self.writer.add_scalar(f"validation/{key}", value, self.trainer.iter)


class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg, writer):
        self.writer = writer  
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, TensorboardLoggerHook(self, self.writer))  
        hooks.insert(-1, ValidationHook(self, self.writer, self.cfg.TEST.EVAL_PERIOD))  
        return hooks


def setup(args):
    """Sets up Detectron2 configuration."""
    cfg = get_cfg()
    
    # Model selection
    model_configs = {
        "mask_rcnn": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "cascade_mask_rcnn": "COCO-InstanceSegmentation/cascade_mask_rcnn_R_50_FPN_3x.yaml",
        "pointrend": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",  # Uses Mask R-CNN base
        "condinst": "COCO-InstanceSegmentation/condinst_R_50_FPN_3x.yaml",
        "solov2": "Cityscapes-InstanceSegmentation/solov2_R50_FPN_3x.yaml"
    }

    if args.instance_model not in model_configs:
        raise ValueError(f"Invalid model name '{args.instance_model}'. Choose from: {list(model_configs.keys())}")

    model_config = model_configs[args.instance_model]
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    
    # Dataset
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # Model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)

    # Training settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Adjust based on dataset
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.epochs
    cfg.TEST.EVAL_PERIOD = 200
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Multi-GPU Support
    cfg.MODEL.DEVICE = "cuda"
    cfg.SOLVER.REFERENCE_WORLD_SIZE = args.num_gpus  

    # Special settings for PointRend
    if args.instance_model == "pointrend":
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.POINT_HEAD.NAME = "StandardPointHead"
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 7  
        cfg.MODEL.POINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    return cfg


def main(args):
    """Main function for training."""
    # Set dataset paths
    train_json = os.path.join(args.data_dir, "annotations", "train.json")
    val_json = os.path.join(args.data_dir, "annotations", "test.json")
    test_json = os.path.join(args.data_dir, "annotations", "val.json")
    train_images = os.path.join(args.data_dir, "train")
    val_images = os.path.join(args.data_dir, "test")
    test_images = os.path.join(args.data_dir, "val")

    # Register dataset in COCO
    register_coco_instances("train", {}, train_json, train_images)
    register_coco_instances("test", {}, val_json, val_images)
    register_coco_instances("val", {}, test_json, test_images)

    cfg = setup(args)
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    trainer = CustomTrainer(cfg, writer)
    trainer.resume_or_load(resume=False)
    trainer.train()
    writer.close()

    # Zip the output directory
    shutil.make_archive(cfg.OUTPUT_DIR, 'zip', cfg.OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 model with multiple GPUs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size per step")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training iterations")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output")
    parser.add_argument("--instance_model", type=str, required=True, 
                        choices=["mask_rcnn", "cascade_mask_rcnn", "pointrend", "condinst", "solov2"],
                        help="Choose instance segmentation model")

    args = parser.parse_args()
    launch(main, args.num_gpus, num_machines=1, machine_rank=0, args=(args,))
