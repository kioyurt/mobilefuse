#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod

import torch

from .mobile_block import MobileConditioningProjector
from .multimodal_llava_encoder.builder import build_vision_tower
from .multimodal_llava_projector.builder import build_vision_projector
from .multimodal_decoder.builder import build_vae, build_sana


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "diffusion_name_or_path"):
            self.dit = build_sana(config)
            self.vae = build_vae(config)
            self.diffusion_connector = MobileConditioningProjector(
                input_dim=896, hidden_dim=512, output_dim=2304,
                num_layers=config.vlm_num_layers,
            )
            self.noise_scheduler = None

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
