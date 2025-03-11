import sys, os
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import wandb
import argparse

class WandbLoggerHook(HookBase):
    def __init__(self, trainer):
        self.trainer = trainer

    def after_step(self):
        storage = self.trainer.storage.latest()
        metrics = {
            "train/loss": storage["total_loss"],
            "train/lr": storage["lr"],
            "train/iteration": self.trainer.iter,
            "train/data_time": storage["data_time"],
            "train/eta_seconds": storage.get("eta_seconds", 0),  # Use .get() to avoid KeyError
            "train/fast_rcnn/cls_accuracy": storage["fast_rcnn/cls_accuracy"],
            "train/loss_box_reg": storage["loss_box_reg"],
            "train/loss_cls": storage["loss_cls"],
            "train/loss_mask": storage["loss_mask"],
            "train/loss_rpn_cls": storage["loss_rpn_cls"],
            "train/loss_rpn_loc": storage["loss_rpn_loc"],
            "train/rank_data_time": storage["rank_data_time"],
            "train/roi_head/num_bg_samples": storage["roi_head/num_bg_samples"],
            "train/roi_head/num_fg_samples": storage["roi_head/num_fg_samples"],
            "train/rpn/num_neg_anchors": storage["rpn/num_neg_anchors"],
            "train/rpn/num_pos_anchors": storage["rpn/num_pos_anchors"]
        }
        wandb.log(metrics)

class ValidationHook(HookBase):
    def __init__(self, trainer, eval_period):
        self.trainer = trainer
        self.eval_period = eval_period

    def after_step(self):
        if self.trainer.iter % self.eval_period == 0 and self.trainer.iter > 0:
            evaluator = COCOEvaluator("test", self.trainer.cfg, False, output_dir=self.trainer.cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.trainer.cfg, "test")
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            wandb.log({f"validation/{key}": value for key, value in results.items()}, step=self.trainer.iter)

class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_hooks([WandbLoggerHook(self), ValidationHook(self, cfg.TEST.EVAL_PERIOD)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 model")
    parser.add_argument("--project_name", type=str, required=True, help="WandB project name")
    parser.add_argument("--batch_sizes", type=int, nargs='+', required=True, help="List of batch sizes to try")
    parser.add_argument("--learning_rates", type=float, nargs='+', required=True, help="List of learning rates to try")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")

    args = parser.parse_args()

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

    for bs in args.batch_sizes:
        for lr in args.learning_rates:
            wandb.init(project=args.project_name)
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = ("train",)
            cfg.DATASETS.TEST = ("test",)
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.SOLVER.STEPS = []
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
            cfg.SOLVER.IMS_PER_BATCH = bs
            cfg.SOLVER.BASE_LR = lr
            cfg.SOLVER.MAX_ITER = 1000
            cfg.TEST.EVAL_PERIOD = 100

            trainer = CustomTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            wandb.finish()
