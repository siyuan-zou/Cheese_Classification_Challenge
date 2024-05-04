from .base import DatasetGenerator
import json
import re
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
    
class ProductionLPromptsGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.type = "production_long"
    def create_prompts(self, labels_names):
        with open("/users/eleves-b/2022/siyuan.zou/DL_SiyuanZou/Chellenge_Cheese/prompts/production_l.json", "r") as f:
            designed_prompts = json.load(f)

        prompts = {}
        for label in labels_names:
            prompts[label] = []
            # remove accents of the label
            if '-' in label:
                # remove spaces around hyphen
                label_cleaned = re.sub(r"\s*-\s*", "-", label)
                # remove accents of the label
                prompts[label].append(
                    {
                        "prompt": designed_prompts[label].replace(f"{label_cleaned.lower()}", f"{label_cleaned.lower()}" + " cheese"),
                        "num_images": self.num_images_per_label,
                    }
                )
            else:
                # remove accents of the label
                prompts[label].append(
                    {
                        "prompt": designed_prompts[label].replace(f"{label.lower()}", f"{label.lower()}" + " cheese"),
                        "num_images": self.num_images_per_label,
                    }
                )

        return prompts

class FoodsPromptsGenerator:
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.type = "foods_prompts"

    def create_prompts(self, labels_names):
        with open("/users/eleves-b/2022/siyuan.zou/DL_SiyuanZou/Chellenge_Cheese/prompts/foods.json", "r") as f:
            designed_prompts = json.load(f)

        prompts = {}
        for label in labels_names:
            prompts[label] = []
            # remove accents of the label
            for food_type in designed_prompts[label]:
                prompts[label].append(
                    {
                        "prompt": label + " cheese:" + food_type,
                        "num_images": self.num_images_per_label,
                    }
                )

        return prompts