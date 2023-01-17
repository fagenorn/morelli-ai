import ast
import base64
import io
import json
import logging
import os
from abc import ABC

import numpy as np
import torch
import transformers
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from captum.attr import IntegratedGradients
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the shared object of compiled Faster Transformer Library if Faster Transformer is set
        if self.setup_config["FasterTransformer"]:
            faster_transformer_complied_path = os.path.join(
                model_dir, "libpyt_fastertransformer.so"
            )
            torch.classes.load_library(faster_transformer_complied_path)
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "image_classification":
                config = AutoConfig.from_pretrained(
                        model_dir,
                        output_attentions=True,
                )
                self.model = AutoModelForImageClassification.from_pretrained(
                        model_dir,
                        config=config,
                )
            else:
                logger.warning("Missing the operation mode.")
            # Using the Better Transformer integration to speedup the inference
            if self.setup_config["BetterTransformer"]:
                try:
                    from optimum.bettertransformer import BetterTransformer

                    self.model = BetterTransformer.transform(self.model)
                except ImportError as error:
                    logger.warning(
                        "HuggingFace Optimum is not installed. Proceeding without BetterTransformer"
                    )
                except RuntimeError as error:
                    logger.warning(
                        "HuggingFace Optimum is not supporting this model,for the list of supported models, please refer to this doc,https://huggingface.co/docs/optimum/bettertransformer/overview"
                    )

            self.model.to(self.device)

        else:
            logger.warning("Missing the checkpoint or state_dict.")



        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
           model_dir,
        )

        if "shortest_edge" in self.feature_extractor.size:
            size = self.feature_extractor.size["shortest_edge"]
        else:
            size = (self.feature_extractor.size["height"], self.feature_extractor.size["width"])

        normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

        def val_transforms(image):
            """Apply _val_transforms across a batch."""
            image = _val_transforms(image.convert("RGB"))
            return image
        
        self.image_processing = val_transforms

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.ig = IntegratedGradients(self.model)

        self.initialized = True

    def preprocess(self, requests):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for idx, data in enumerate(requests):
            image = data.get("data") or data.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return self.feature_extractor(images=images, return_tensors="pt").to(self.device)

    def inference(self, data):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """

        with torch.no_grad():
            results = self.model(**data)

        return results

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """

        result = []

        num_rows = inference_output.logits.shape[0]
        for i in range(num_rows):
            logits = inference_output.logits[i].unsqueeze(0)
            attentions = inference_output.attentions[i].unsqueeze(0)
            predicted_label = logits.argmax(-1).item()
            label = self.model.config.id2label[predicted_label]
            probs = torch.softmax(logits, dim=1)
            prob = round(probs[0][0].item(), 4)

            result.append({
                "label": label,
                "ai_chance": prob,
            })

        return result

    def get_insights(self, batch_feature, _, target=0):
        """This function initialize and calls the layer integrated gradient to get word importance
        of the input text if captum explanation has been selected through setup_config
        Args:
            input_batch (int): Batches of tokens IDs of text
            text (str): The Text specified in the input request
            target (int): The Target can be set to any acceptable label under the user's discretion.
        Returns:
            (list): Returns a list of importances and words.
        """

        tensor_data =batch_feature.convert_to_tensors(tensor_type="pt")
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)

    return mask   