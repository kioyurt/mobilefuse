# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from ..llava_arch import (
    LlavaMetaModel,
    LlavaMetaForCausalLM,
)

# ============================================================
# 原始 Mobile-O Projector
# ============================================================

from ..mobile_block import (
    MobileConditioningProjector,
)

# ============================================================
# 新的 QSCFMCP
# ============================================================

from ..qscfmcp import (
    QSCFMCP,
)


# ============================================================
# Config
# ============================================================


class mobileoFastConfig(Qwen2Config):

    model_type = "llava_qwen2"

    def __init__(
        self,
        *args,
        use_qscfmcp=False,
        qscfmcp_input_dim=2048,
        qscfmcp_hidden_dim=512,
        qscfmcp_output_dim=2304,
        qscfmcp_num_layers=8,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        # --------------------------------------------------------
        # QSCFMCP configuration
        # --------------------------------------------------------

        self.use_qscfmcp = use_qscfmcp

        self.qscfmcp_input_dim = (
            qscfmcp_input_dim
        )

        self.qscfmcp_hidden_dim = (
            qscfmcp_hidden_dim
        )

        self.qscfmcp_output_dim = (
            qscfmcp_output_dim
        )

        self.qscfmcp_num_layers = (
            qscfmcp_num_layers
        )


# ============================================================
# Mobile-O Model
# ============================================================


class mobileoFastModel(
    LlavaMetaModel,
    Qwen2Model,
):

    config_class = mobileoFastConfig

    def __init__(
        self,
        config: Qwen2Config,
    ):

        super(
            mobileoFastModel,
            self
        ).__init__(
            config
        )

        # --------------------------------------------------------
        # 原始 Mobile-O connector
        #
        # 默认仍然保留。
        #
        # --------------------------------------------------------

        if not getattr(
            config,
            "use_qscfmcp",
            False,
        ):

            if not hasattr(
                self,
                "diffusion_connector",
            ):

                self.diffusion_connector = (
                    MobileConditioningProjector(
                        input_dim=getattr(
                            config,
                            "hidden_size",
                            896,
                        ),
                        hidden_dim=512,
                        output_dim=2304,
                        num_layers=getattr(
                            config,
                            "vlm_num_layers",
                            8,
                        ),
                    )
                )

        # --------------------------------------------------------
        # 新 QSCFMCP
        #
        # Qwen2.5-VL:
        # hidden = 2048
        # layers = 36
        #
        # 我们使用最后 8 层。
        #
        # --------------------------------------------------------

        else:

            self.diffusion_connector = (
                QSCFMCP(
                    input_dim=(
                        config.qscfmcp_input_dim
                    ),
                    hidden_dim=(
                        config.qscfmcp_hidden_dim
                    ),
                    output_dim=(
                        config.qscfmcp_output_dim
                    ),
                    num_layers=(
                        config.qscfmcp_num_layers
                    ),
                )
            )


# ============================================================
# Mobile-O Causal LM
# ============================================================


class mobileoFastForCausalLM(
    Qwen2ForCausalLM,
    LlavaMetaForCausalLM,
):

    config_class = mobileoFastConfig

    def __init__(
        self,
        config,
    ):

        super(
            Qwen2ForCausalLM,
            self
        ).__init__(
            config
        )

        self.model = (
            mobileoFastModel(
                config
            )
        )

        self.vocab_size = (
            config.vocab_size
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()


    def get_model(self):

        return self.model


    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[
            torch.Tensor
        ] = None,

        position_ids: Optional[
            torch.LongTensor
        ] = None,

        past_key_values: Optional[
            List[torch.FloatTensor]
        ] = None,

        inputs_embeds: Optional[
            torch.FloatTensor
        ] = None,

        labels: Optional[
            torch.LongTensor
        ] = None,

        use_cache: Optional[
            bool
        ] = None,

        output_attentions: Optional[
            bool
        ] = None,

        output_hidden_states: Optional[
            bool
        ] = None,

        gen_image: Optional[
            torch.FloatTensor
        ] = None,

        und_image: Optional[
            torch.FloatTensor
        ] = None,

        categories: Optional[
            List[str]
        ] = None,

        return_dict: Optional[
            bool
        ] = None,

        cache_position: Optional[
            torch.LongTensor
        ] = None,

        # ========================================================
        # 新增参数
        #
        # 这两个参数来自 Qwen2.5-VL wrapper。
        # ========================================================

        visible_hidden_states: Optional[
            List[torch.Tensor]
        ] = None,

        infrared_hidden_states: Optional[
            List[torch.Tensor]
        ] = None,

    ) -> Union[
        Tuple,
        CausalLMOutputWithPast,
    ]:

        # --------------------------------------------------------
        # 1. Attention settings
        # --------------------------------------------------------

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        # --------------------------------------------------------
        # 我们必须获得所有 hidden states
        # --------------------------------------------------------

        output_hidden_states = True

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )


        # --------------------------------------------------------
        # 2. Multimodal preparation
        # --------------------------------------------------------

        if inputs_embeds is None:

            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latents,
            ) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    gen_image,
                    und_image,
                )
            )

        else:

            # ----------------------------------------------------
            # 如果外部直接传 inputs_embeds，
            # 原始 Mobile-O 流程不会产生 latents。
            #
            # 为保持逻辑清晰，这里初始化。
            # ----------------------------------------------------

            latents = None


        # --------------------------------------------------------
        # 3. Language model forward
        # --------------------------------------------------------

        output = super().forward(

            input_ids=input_ids,

            attention_mask=attention_mask,

            position_ids=position_ids,

            past_key_values=past_key_values,

            inputs_embeds=inputs_embeds,

            labels=labels,

            use_cache=use_cache,

            output_attentions=output_attentions,

            output_hidden_states=True,

            return_dict=return_dict,

        )


        # --------------------------------------------------------
        # 4. Outputs
        # --------------------------------------------------------

        ce_loss = output.loss

        hidden_states = (
            output.hidden_states
        )

        logits = output.logits


        # --------------------------------------------------------
        # 原始 Mobile-O：
        #
        # img_hidden_states = hidden_states
        #
        # 这里继续保留。
        # --------------------------------------------------------

        img_hidden_states = (
            hidden_states
        )


        # --------------------------------------------------------
        # 5. Latents check
        # --------------------------------------------------------

        assert (
            latents is not None
        ), (
            "Currently we only support image "
            "loss when latents is not None"
        )


        # ========================================================
        # 6. Diffusion timestep sampling
        # ========================================================

        weighting_scheme = "uniform"


        u = (
            compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,

                batch_size=(
                    latents.shape[0]
                ),

                logit_mean=0.0,

                logit_std=1.0,

                mode_scale=1.29,
            )
        )


        indices = (
            u
            * self.get_model()
            .noise_scheduler
            .config
            .num_train_timesteps
        ).long()


        timesteps = (
            self.get_model()
            .noise_scheduler
            .timesteps[indices]
            .to(
                device=latents.device
            )
        )


        sigmas = (
            self.get_sigmas(
                timesteps,
                latents.device,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
        )


        # ========================================================
        # 7. Add noise
        # ========================================================

        noise = torch.randn_like(
            latents,
            device=latents.device,
        )


        noisy_latents = (
            (1.0 - sigmas)
            * latents
            +
            sigmas
            * noise
        )


        # ========================================================
        # 8. Generate diffusion condition
        # ========================================================

        use_qscfmcp = getattr(
            self.config,
            "use_qscfmcp",
            False,
        )


        # ========================================================
        # 新路径
        #
        # Qwen2.5-VL:
        #
        # Visible hidden
        # Infrared hidden
        #
        #       ↓
        #
        # QSCFMCP
        #
        #       ↓
        #
        # [B,Nv+Ni,2304]
        #
        # ========================================================

        if use_qscfmcp:

            if (
                visible_hidden_states
                is None
            ):

                raise RuntimeError(
                    "use_qscfmcp=True, but "
                    "visible_hidden_states is None."
                )


            if (
                infrared_hidden_states
                is None
            ):

                raise RuntimeError(
                    "use_qscfmcp=True, but "
                    "infrared_hidden_states is None."
                )


            condition = (
                self.get_model()
                .diffusion_connector(
                    visible_hidden_states,
                    infrared_hidden_states,
                )
            )


            # ----------------------------------------------------
            # QSCFMCP 只处理 visual tokens。
            #
            # 所以它的 attention mask 不能继续使用
            # 原始 LLM attention_mask。
            #
            # 原始：
            #
            # [B, sequence_length]
            #
            # 新：
            #
            # [B, Nv + Nir]
            #
            # ----------------------------------------------------

            condition_attention_mask = (
                torch.ones(
                    condition.shape[
                        0
                    ],
                    condition.shape[
                        1
                    ],
                    dtype=torch.bool,
                    device=condition.device,
                )
            )


        # ========================================================
        # 原 Mobile-O 路径
        #
        # 完全保留。
        #
        # ========================================================

        else:

            condition = (
                self.get_model()
                .diffusion_connector(
                    img_hidden_states
                )
            )


            condition_attention_mask = (
                attention_mask
            )


        # ========================================================
        # 9. SANA DiT
        # ========================================================

        diffusion_pred = (
            self.get_model()
            .dit(

                hidden_states=(
                    noisy_latents
                ),

                timestep=(
                    timesteps
                ),

                encoder_hidden_states=(
                    condition
                ),

                encoder_attention_mask=(
                    condition_attention_mask
                ),

            )
            .sample
        )


        # ========================================================
        # 10. Diffusion target
        # ========================================================

        target = (
            noise - latents
        )


        weighting = (
            compute_loss_weighting_for_sd3(
                weighting_scheme=(
                    weighting_scheme
                ),
                sigmas=sigmas,
            )
        )


        diff_loss = torch.mean(

            (
                weighting.float()
                *
                (
                    diffusion_pred.float()
                    -
                    target.float()
                )
                ** 2
            )
            .reshape(
                target.shape[0],
                -1,
            ),

            1,
        )


        diff_loss = (
            diff_loss.mean()
        )


        # ========================================================
        # 11. Loss weighting
        # ========================================================

        ce_weight = 0.2

        diff_weight = 1.0


        total_loss = (
            ce_weight * ce_loss
            +
            diff_weight * diff_loss
        )


        print(
            f"diff_loss: {diff_loss}, "
            f"ce_loss: {ce_loss}"
        )


        # ========================================================
        # 12. Return
        # ========================================================

        return CausalLMOutputWithPast(

            loss=total_loss,

            logits=logits,

            past_key_values=(
                output.past_key_values
            ),

            hidden_states=(
                output.hidden_states
            ),

            attentions=(
                output.attentions
            ),

        )


# ================================================================
# Register
# ================================================================

AutoConfig.register(
    "llava_qwen2",
    mobileoFastConfig,
)


AutoModelForCausalLM.register(
    mobileoFastConfig,
    mobileoFastForCausalLM,
)