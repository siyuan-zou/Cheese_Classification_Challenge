from .base import DatasetGenerator
import json
# from unidecode import unidecode

class SimplePromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.type = "simple_prompts"

    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            prompts[label] = []
            prompts[label].append(
                {
                    "prompt": f"An image of a {label} cheese",
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts

class SellingPromptsGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.type = "selling_prompts"
    def create_prompts(self, labels_names):
        with open("/users/eleves-b/2022/siyuan.zou/DL_SiyuanZou/Chellenge_Cheese/prompts/selling.json", "r") as f:
            designed_prompts = json.load(f)

        prompts = {}
        for label in labels_names:
            prompts[label] = []
            # remove accents of the label
            prompts[label].append(
                {
                    "prompt": designed_prompts[label].replace(f"{label.lower()}", f"{label.lower()}" + " cheese"),
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts
    
class ProductionPromptsGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.type = "production_prompts"
    def create_prompts(self, labels_names):
        with open("/users/eleves-b/2022/siyuan.zou/DL_SiyuanZou/Chellenge_Cheese/prompts/production.json", "r") as f:
            designed_prompts = json.load(f)

        prompts = {}
        for label in labels_names:
            prompts[label] = []
            # remove accents of the label
            prompts[label].append(
                {
                    "prompt": designed_prompts[label].replace(f"{label.lower()}", f"{label.lower()}" + " cheese"),
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts