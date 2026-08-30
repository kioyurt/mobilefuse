# -*- coding:utf-8 -*-

"""
Head-specific Semantic Attention Bias Generator


Input:

    semantic_tokens

        from:

        SAM + Qwen full hidden


        [B,Ns,C]


    latent_tokens

        SANA latent query


        [B,Nq,C]


    condition_tokens

        SANA encoder condition


        [B,Nk,C]



Output:


    semantic_bias


        [B,num_heads,Nq,Nk]



Usage:


    attn_score += semantic_bias



"""


from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F





class SpatialSemanticProjector(nn.Module):
    """
    Convert SAM-Qwen semantic tokens
    into SANA spatial resolution.


    Because:


    SAM:

        64x64


    SANA latent:

        32x32


    Need alignment.



    """



    def __init__(
        self,
        dim=1024
    ):

        super().__init__()



        self.proj = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.GELU(),

            nn.LayerNorm(
                dim
            )

        )





    def forward(
        self,
        semantic_tokens,
        target_length
    ):


        """

        Args:


        semantic_tokens:

            [B,N,C]



        target_length:

            Nq


        """


        B,N,C = semantic_tokens.shape



        if N == target_length:

            return self.proj(
                semantic_tokens
            )



        # -----------------------------
        # token interpolation
        # -----------------------------


        x = semantic_tokens.transpose(
            1,
            2
        )


        x = F.interpolate(

            x,

            size=target_length,

            mode="linear",

            align_corners=False

        )


        x = x.transpose(
            1,
            2
        )


        return self.proj(
            x
        )










class HeadSpecificSemanticBiasGenerator(
    nn.Module
):


    """

    Generate:

        head-specific attention bias


    Equation:


    B_h = Q_s K_c^T



    where:


        Q_s:

            SAM-Qwen semantic spatial query


        K_c:

            SANA condition key



    """



    def __init__(

        self,

        dim=1024,

        num_heads=16,

        bias_scale=1.0

    ):

        super().__init__()



        self.dim = dim

        self.num_heads=num_heads


        self.head_dim = (
            dim //
            num_heads
        )


        assert (
            dim % num_heads ==0
        )



        self.bias_scale = bias_scale



        # ----------------------------------
        # semantic token projection
        # ----------------------------------


        self.semantic_q = nn.Linear(

            dim,

            dim

        )



        # ----------------------------------
        # condition projection
        # ----------------------------------


        self.condition_k = nn.Linear(

            dim,

            dim

        )



        # ----------------------------------
        # head adaptive gate
        # ----------------------------------


        self.head_gate = nn.Sequential(

            nn.Linear(

                dim,

                num_heads

            ),

            nn.Sigmoid()

        )



        # ----------------------------------
        # normalize
        # ----------------------------------


        self.norm = nn.LayerNorm(
            dim
        )





    def forward(

        self,

        semantic_tokens,

        condition_tokens,

        latent_length=None

    ):


        """

        Args:



        semantic_tokens:

            SAM-Qwen semantic feature


            [B,Ns,C]



        condition_tokens:

            SANA encoder hidden


            [B,Nk,C]



        latent_length:


            Nq



        Returns:


            bias:


            [B,H,Nq,Nk]


        """



        semantic_tokens = self.norm(
            semantic_tokens
        )


        condition_tokens = self.norm(
            condition_tokens
        )



        # =================================
        # 1. spatial alignment
        # =================================


        if latent_length is not None:


            semantic_tokens = (
                SpatialSemanticProjector(
                    semantic_tokens.shape[-1]
                )
                (
                    semantic_tokens,
                    latent_length
                )
            )



        # =================================
        # 2. Semantic Query
        # =================================


        q = self.semantic_q(
            semantic_tokens
        )



        # [B,Nq,C]



        # =================================
        # 3. Condition Key
        # =================================


        k = self.condition_k(
            condition_tokens
        )



        # [B,Nk,C]



        B,Nq,C = q.shape

        Nk = k.shape[1]



        # =================================
        # 4. split heads
        # =================================


        q = q.reshape(

            B,

            Nq,

            self.num_heads,

            self.head_dim

        )


        q = q.permute(

            0,

            2,

            1,

            3

        )



        # [B,H,Nq,D]



        k = k.reshape(

            B,

            Nk,

            self.num_heads,

            self.head_dim

        )


        k = k.permute(

            0,

            2,

            1,

            3

        )



        # [B,H,Nk,D]



        # =================================
        # 5. attention bias
        # =================================


        bias = torch.matmul(

            q,

            k.transpose(
                -1,
                -2
            )

        )


        bias = (
            bias /
            self.head_dim**0.5
        )



        # =================================
        # 6. head-specific gate
        # =================================


        gate = self.head_gate(
            semantic_tokens.mean(
                dim=1
            )
        )


        # [B,H]



        gate = gate.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )


        bias = (

            bias
            *
            gate

        )



        bias = (

            bias
            *
            self.bias_scale

        )



        return bias