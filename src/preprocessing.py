import os
import numpy as np
import json
import itertools
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

class PrepProcessDataset:

    def get_building_dicts(self, img_dir):
        """
           Convert the original dataset into the Dectron2 format
        """
        json_file = os.path.join(img_dir, "annotations.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns.values()):
            record = {}
            
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            annos = v["regions"]
            objs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = list(itertools.chain.from_iterable(poly))

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    def create_dataset(self):
        for d in ["train", "val"]:
            DatasetCatalog.register("buildings_" + d, lambda d=d: self.get_building_dicts("buildings/" + d))
            MetadataCatalog.get("buildings_" + d).set(thing_classes=["buildings"])
        building_metadata = MetadataCatalog.get("buildings_train")
        return building_metadata

    def visualize_samples(self, building_metadata):
        dataset_dicts = self.get_building_dicts("buildings/train")
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2_imshow(vis.get_image()[:, :, ::-1])

if __name__ == '__main__':
    preprocess_obj = PrepProcessDataset()
    # Prepare train and val dataset
    building_metadata = preprocess_obj.create_dataset()
    # to visulize a few samples
    preprocess_obj.visualize_samples(building_metadata)