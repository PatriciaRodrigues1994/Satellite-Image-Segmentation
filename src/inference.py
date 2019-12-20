from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg

class Inference():
	def get_trained_model(self):
		cfg = get_cfg()
		cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
		cfg.DATASETS.TEST = ("train_test", )
		predictor = DefaultPredictor(cfg)


	def get_predictions(self, predictor, building_metadata):
		dataset_dicts = get_balloon_dicts("building/test")
		building_metadata = MetadataCatalog.get("buildings_test")
		for d in random.sample(dataset_dicts, 3):    
		    im = cv2.imread(d["file_name"])
		    outputs = predictor(im)
		    v = Visualizer(im[:, :, ::-1],
		                   metadata=building_metadata, 
		                   scale=0.8, 
		                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
		    )
		    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		    cv2_imshow(v.get_image()[:, :, ::-1])


if __name__ == '__main__':
	infer_obj = Inference()
	predictor = infer_obj.get_trained_model()
	infer_obj.get_predictions(predictor)

