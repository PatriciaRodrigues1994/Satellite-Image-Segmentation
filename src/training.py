from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


class Trainer:
	def create_config(self):
		cfg = get_cfg()
		cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		cfg.DATASETS.TRAIN = ("buildings_train",)
		cfg.DATASETS.TEST = ()
		cfg.DATALOADER.NUM_WORKERS = 2
		cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
		cfg.SOLVER.IMS_PER_BATCH = 2
		cfg.SOLVER.BASE_LR = 0.00025
		cfg.SOLVER.MAX_ITER = 3000    
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (building)
		return cfg

	def train_model(self, cgf):
		os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
		trainer = DefaultTrainer(cfg) 
		trainer.resume_or_load(resume=False)
		trainer.train()

if __name__ == '__main__':
	train_obj = Trainer()
	cfg = train_obj.create_config()
	train_obj.train_model(cfg)