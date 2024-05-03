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

    def create_prompts(self, labels_names):
        with open("prompts/selling.json", "r") as f:
            selling_prompts = json.load(f)

        prompts = {}
        for label in labels_names:
            prompts[label] = []
            # remove accents of the label
            prompts[label].append(
                {
                    "prompt": selling_prompts[label],
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts