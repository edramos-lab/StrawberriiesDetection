
import sys, os
# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import torch, detectron2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import os


# Set dataset paths
data_dir = "/content/Strawberry-Diseases-1"
train_json = os.path.join(data_dir, "annotations", "train.json")
val_json = os.path.join(data_dir, "annotations", "test.json")
test_json = os.path.join(data_dir, "annotations", "val.json")
train_images = os.path.join(data_dir, "train")
val_images = os.path.join(data_dir, "test")
test_images = os.path.join(data_dir, "val")

# Register dataset in COCO
from detectron2.data.datasets import register_coco_instances

register_coco_instances("/content/Strawberry-Diseases-1/train", {}, train_json, train_images)
register_coco_instances("/content/Strawberry-Diseases-1/test", {}, val_json, val_images)
register_coco_instances("/content/Strawberry-Diseases-1/val", {}, test_json, test_images)

"""To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:


"""

from detectron2.data import MetadataCatalog, DatasetCatalog

dataset_dicts =  DatasetCatalog.get("/content/Strawberry-Diseases-1/train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1],scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Sample", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""## Train!

Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~2 minutes to train 300 iterations on a P100 GPU.

"""



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("/content/Strawberry-Diseases-1/train",)
cfg.DATASETS.TEST = ("/content/Strawberry-Diseases-1/test",) #set test to val
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # only has one class of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.SOLVER.IMS_PER_BATCH = 32  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR =0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000
cfg.TEST.EVAL_PERIOD = 100  # Validate every 200 iterations

import wandb
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.data import build_detection_test_loader
import argparse
class WandbLoggerHook(HookBase):
        def __init__(self, trainer):
            self.trainer = trainer

        def after_step(self):
            # Log every 100 iterations
            metrics = {
                "train/loss": self.trainer.storage.latest()["total_loss"],
                "train/lr": self.trainer.storage.latest()["lr"],
                "train/iteration": self.trainer.iter,
                "train/data_time": self.trainer.storage.latest()["data_time"],
                "train/eta_seconds": self.trainer.storage.latest()["eta_seconds"],
                "train/fast_rcnn/cls_accuracy": self.trainer.storage.latest()["fast_rcnn/cls_accuracy"],
                "train/loss_box_reg": self.trainer.storage.latest()["loss_box_reg"],
                "train/loss_cls": self.trainer.storage.latest()["loss_cls"],
                "train/loss_mask": self.trainer.storage.latest()["loss_mask"],
                "train/loss_rpn_cls": self.trainer.storage.latest()["loss_rpn_cls"],
                "train/loss_rpn_loc": self.trainer.storage.latest()["loss_rpn_loc"],
                "train/rank_data_time": self.trainer.storage.latest()["rank_data_time"],
                "train/roi_head/num_bg_samples": self.trainer.storage.latest()["roi_head/num_bg_samples"],
                "train/roi_head/num_fg_samples": self.trainer.storage.latest()["roi_head/num_fg_samples"],
                "train/rpn/num_neg_anchors": self.trainer.storage.latest()["rpn/num_neg_anchors"],
                "train/rpn/num_pos_anchors": self.trainer.storage.latest()["rpn/num_pos_anchors"],
                "train/time": self.trainer.storage.latest()["time"]
            }
            wandb.log(metrics)
                

class ValidationHook(HookBase):
    def __init__(self, trainer, eval_period):
        self.trainer = trainer
        self.eval_period = eval_period

    def after_step(self):
        if self.trainer.iter % self.eval_period == 0 and self.trainer.iter > 0:
            evaluator = COCOEvaluator("/content/Strawberry-Diseases-1/test", self.trainer.cfg, False, output_dir=self.trainer.cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.trainer.cfg, "/content/Strawberry-Diseases-1/test")
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

            # Log validation metrics in WandB
            #wandb.log({"validation/AP": results["bbox"]["AP"], "iteration": self.trainer.iter})
            # Log all validation metrics in WandB
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
            # Initialize WandB
            wandb.init(project=args.project_name)
            cfg.SOLVER.IMS_PER_BATCH = bs  # This is the real "batch size" commonly known to deep learning people
            cfg.SOLVER.BASE_LR = lr  # pick a good LR
            cfg.TEST.EVAL_PERIOD = 100  # Validate every 100 iterations
            trainer = CustomTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            wandb.finish()
