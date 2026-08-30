# magic.py

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



from diffusers import (
    AutoencoderDC,
    SanaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler
)



# ============================================================
# Utils
# ============================================================


def zero_module(module):

    for p in module.parameters():
        nn.init.zeros_(p)

    return module



# ============================================================
# RMSNorm
# ============================================================


class RMSNorm(nn.Module):

    def __init__(
        self,
        dim,
        eps=1e-6
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps



    def forward(self,x):

        norm = torch.rsqrt(
            x.pow(2)
            .mean(
                dim=-1,
                keepdim=True
            )
            +
            self.eps
        )

        return x * norm * self.weight



# ============================================================
# Timestep embedding
# ============================================================


class TimestepEmbedding(nn.Module):

    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.dim=dim


        self.proj=nn.Sequential(

            nn.Linear(dim,dim*4),

            nn.SiLU(),

            nn.Linear(dim*4,dim)

        )


    def forward(self,t):

        half=self.dim//2


        freq=torch.exp(
            -math.log(10000)
            *
            torch.arange(
                half,
                device=t.device
            )
            /
            half
        )


        args=t[:,None]*freq[None]


        emb=torch.cat(
            [
                torch.sin(args),
                torch.cos(args)
            ],
            dim=-1
        )


        return self.proj(emb)




# ============================================================
# Ada Layer Norm
# ============================================================


class AdaLayerNorm(nn.Module):


    def __init__(
        self,
        dim,
        cond_dim
    ):
        super().__init__()


        self.norm=nn.LayerNorm(
            dim,
            elementwise_affine=False
        )


        self.modulation=nn.Sequential(

            nn.SiLU(),

            nn.Linear(
                cond_dim,
                dim*2
            )

        )



    def forward(
        self,
        x,
        cond
    ):

        h=self.norm(x)


        scale,shift=self.modulation(cond).chunk(
            2,
            dim=-1
        )


        return h*(1+scale.unsqueeze(1))+shift.unsqueeze(1)




# ============================================================
# Cross Attention
# ============================================================


class CrossAttention(nn.Module):


    def __init__(
        self,
        dim,
        context_dim,
        heads=8,
        head_dim=64
    ):
        super().__init__()


        self.heads=heads

        inner=heads*head_dim


        self.scale=head_dim**-0.5


        self.to_q=nn.Linear(
            dim,
            inner,
            bias=False
        )


        self.to_k=nn.Linear(
            context_dim,
            inner,
            bias=False
        )


        self.to_v=nn.Linear(
            context_dim,
            inner,
            bias=False
        )


        self.out=nn.Linear(
            inner,
            dim
        )



    def forward(
        self,
        x,
        context
    ):


        B,N,C=x.shape


        q=self.to_q(x)

        k=self.to_k(context)

        v=self.to_v(context)



        q=q.reshape(
            B,
            N,
            self.heads,
            -1
        ).transpose(1,2)


        k=k.reshape(
            B,
            -1,
            self.heads,
            q.shape[-1]
        ).transpose(1,2)



        v=v.reshape(
            B,
            -1,
            self.heads,
            q.shape[-1]
        ).transpose(1,2)



        attn=torch.matmul(
            q,
            k.transpose(-1,-2)
        )

        attn=attn*self.scale


        attn=attn.softmax(
            dim=-1
        )


        out=torch.matmul(
            attn,
            v
        )


        out=out.transpose(
            1,2
        ).reshape(
            B,N,-1
        )


        return self.out(out)




# ============================================================
# Feed Forward
# ============================================================


class FeedForward(nn.Module):

    def __init__(
        self,
        dim,
        mult=4
    ):
        super().__init__()


        self.net=nn.Sequential(

            nn.Linear(
                dim,
                dim*mult
            ),

            nn.GELU(),

            nn.Linear(
                dim*mult,
                dim
            )

        )



    def forward(self,x):

        return self.net(x)




# ============================================================
# IKR Adapter
# ============================================================


class IKRAdapter(nn.Module):

    """
    Intra-spectral Knowledge Restoration


    eps_ikr =
        eps_sana
        +
        delta_ikr


    """


    def __init__(
        self,
        latent_dim,
        cond_dim,
        hidden_dim=512
    ):
        super().__init__()



        self.latent_proj=nn.Linear(
            latent_dim,
            hidden_dim
        )


        self.cond_attn=CrossAttention(
            hidden_dim,
            cond_dim
        )


        self.time_embed=TimestepEmbedding(
            hidden_dim
        )


        self.norm=AdaLayerNorm(
            hidden_dim,
            hidden_dim
        )


        self.ffn=FeedForward(
            hidden_dim
        )


        self.out=zero_module(
            nn.Linear(
                hidden_dim,
                latent_dim
            )
        )



    def forward(
        self,
        latent,
        cond,
        timestep
    ):


        # latent:
        # B,C,H,W


        B,C,H,W=latent.shape


        x=latent.flatten(
            2
        ).transpose(
            1,2
        )


        x=self.latent_proj(x)


        x=self.cond_attn(
            x,
            cond
        )


        t=self.time_embed(
            timestep
        )


        x=self.norm(
            x,
            t
        )


        x=self.ffn(x)


        out=self.out(x)


        out=out.transpose(
            1,2
        ).reshape(
            B,
            C,
            H,
            W
        )


        return out




# ============================================================
# CKG Adapter
# ============================================================


class CKGAdapter(nn.Module):


    """
    Cross Spectral Knowledge Generation


    visible condition
           +
    thermal memory


    generate infrared residual


    """


    def __init__(
        self,
        latent_dim,
        cond_dim,
        memory_dim,
        hidden_dim=512
    ):
        super().__init__()



        self.latent_proj=nn.Linear(
            latent_dim,
            hidden_dim
        )


        self.visible_attn=CrossAttention(
            hidden_dim,
            cond_dim
        )


        self.thermal_attn=CrossAttention(
            hidden_dim,
            memory_dim
        )



        self.time_embed=TimestepEmbedding(
            hidden_dim
        )



        self.norm=AdaLayerNorm(
            hidden_dim,
            hidden_dim
        )


        self.ffn=FeedForward(
            hidden_dim
        )



        self.out=zero_module(
            nn.Linear(
                hidden_dim,
                latent_dim
            )
        )



    def forward(
        self,
        latent,
        visible_cond,
        thermal_memory,
        timestep
    ):


        B,C,H,W=latent.shape


        x=latent.flatten(
            2
        ).transpose(
            1,2
        )


        x=self.latent_proj(x)



        x=x+self.visible_attn(
            x,
            visible_cond
        )



        x=x+self.thermal_attn(
            x,
            thermal_memory
        )



        t=self.time_embed(
            timestep
        )


        x=self.norm(
            x,
            t
        )


        x=self.ffn(x)


        out=self.out(x)


        out=out.transpose(
            1,2
        ).reshape(
            B,C,H,W
        )


        return out

# ============================================================
# Missing Modality Encoder
# ============================================================


class MissingModalityEncoder(nn.Module):

    """
    Missing infrared modality controller

    当IR缺失时提供thermal prior memory
    """

    def __init__(
        self,
        dim=512,
        memory_tokens=32
    ):
        super().__init__()


        self.memory_tokens=memory_tokens


        self.memory_bank=nn.Parameter(
            torch.randn(
                memory_tokens,
                dim
            )
        )


        self.mask_embedding=nn.Embedding(
            2,
            dim
        )


        nn.init.normal_(
            self.memory_bank,
            std=0.02
        )


    def forward(
        self,
        batch_size,
        modality_mask,
        device
    ):


        if modality_mask is None:

            modality_mask=torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device
            )


        mask_token=self.mask_embedding(
            modality_mask
        )


        memory=self.memory_bank.unsqueeze(0)


        memory=memory.repeat(
            batch_size,
            1,
            1
        )


        memory=memory+mask_token[:,None,:]


        return memory



# ============================================================
# MKF Transformer Noise Fusion
# ============================================================


class MKFTransformerFusion(nn.Module):

    """
    Multi-domain Knowledge Fusion


    epsilon_f =
        w_t * epsilon_ikr
        +
        (1-w_t)*epsilon_ckg


    weight depends on:

    x0_ikr
    x0_ckg
    zt
    timestep


    """


    def __init__(
        self,
        latent_dim=32,
        hidden_dim=512,
        heads=8,
        layers=4
    ):
        super().__init__()



        self.input_proj=nn.Linear(
            latent_dim*5,
            hidden_dim
        )


        encoder_layer=nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )


        self.transformer=nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )


        self.time_embed=TimestepEmbedding(
            hidden_dim
        )


        self.weight_head=nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                1
            ),

            nn.Sigmoid()

        )



    def forward(
        self,
        zt,
        eps_ikr,
        eps_ckg,
        timestep
    ):


        """
        zt:
            current noisy latent

        eps_ikr:
            RGB restoration noise

        eps_ckg:
            IR generation noise

        """


        # estimate clean latent

        x0_ikr = zt - eps_ikr

        x0_ckg = zt - eps_ckg



        B,C,H,W=zt.shape


        tokens=torch.cat(
            [
                zt,
                x0_ikr,
                x0_ckg,
                eps_ikr,
                eps_ckg
            ],
            dim=1
        )


        tokens=tokens.flatten(
            2
        ).transpose(
            1,
            2
        )


        tokens=self.input_proj(
            tokens
        )



        t=self.time_embed(
            timestep
        )


        tokens=tokens+t[:,None,:]



        tokens=self.transformer(
            tokens
        )


        global_feature=tokens.mean(
            dim=1
        )


        weight=self.weight_head(
            global_feature
        )


        weight=weight[:,:,None,None]


        eps_fused=(

            weight*eps_ikr

            +

            (1-weight)*eps_ckg

        )


        return eps_fused,weight



# ============================================================
# MagicFuse SANA Wrapper
# ============================================================


class MagicFuseSANAWrapper(nn.Module):


    def __init__(
        self,
        sana_path,
        dtype=torch.float16,
        latent_dim=32,
        cond_dim=2304
    ):

        super().__init__()



        print(
            "Loading SANA backbone..."
        )


        self.dit=SanaTransformer2DModel.from_pretrained(
            sana_path,
            subfolder="transformer",
            torch_dtype=dtype
        )


        print(
            "Loading DC-AE..."
        )


        self.vae=AutoencoderDC.from_pretrained(
            sana_path,
            subfolder="vae",
            torch_dtype=dtype
        )



        self.noise_scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            sana_path,
            subfolder="scheduler"
        )



        # ----------------------------------------------------
        # Freeze pretrained modules
        # ----------------------------------------------------


        for p in self.dit.parameters():

            p.requires_grad=False


        for p in self.vae.parameters():

            p.requires_grad=False



        self.dit.eval()

        self.vae.eval()



        # ----------------------------------------------------
        # MagicFuse branches
        # ----------------------------------------------------


        self.branch_a=IKRAdapter(
            latent_dim=latent_dim,
            cond_dim=cond_dim
        )


        self.branch_b=CKGAdapter(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            memory_dim=512
        )


        self.branch_c=MKFTransformerFusion(
            latent_dim=latent_dim
        )



        self.missing_mod=MissingModalityEncoder(
            dim=512
        )



    # ========================================================
    # VAE encode
    # ========================================================


    @torch.no_grad()
    def encode_image(
        self,
        image
    ):


        posterior=self.vae.encode(
            image
        )


        latent=posterior.latent


        return latent



    # ========================================================
    # VAE decode
    # ========================================================


    @torch.no_grad()
    def decode_latent(
        self,
        latent
    ):


        image=self.vae.decode(
            latent
        ).sample


        return image



    # ========================================================
    # Diffusion forward
    # ========================================================


    def forward(
        self,
        noisy_latent,
        static_cond,
        timesteps,
        modality_mask=None,
        attention_mask=None
    ):


        B=noisy_latent.shape[0]



        # ================================
        # Thermal memory
        # ================================


        thermal_memory=self.missing_mod(
            B,
            modality_mask,
            noisy_latent.device
        )



        # ================================
        # Base SANA prediction
        # ================================


        with torch.no_grad():

            sana_eps=self.dit(
                hidden_states=noisy_latent,
                timestep=timesteps,
                encoder_hidden_states=static_cond,
                encoder_attention_mask=attention_mask
            ).sample



        # ================================
        # IKR
        # ================================


        ikr_delta=self.branch_a(
            noisy_latent,
            static_cond,
            timesteps
        )


        eps_ikr=sana_eps+ikr_delta



        # ================================
        # CKG
        # ================================


        ckg_delta=self.branch_b(
            noisy_latent,
            static_cond,
            thermal_memory,
            timesteps
        )


        eps_ckg=sana_eps+ckg_delta



        # ================================
        # MKF
        # ================================


        eps_fused,weight=self.branch_c(
            noisy_latent,
            eps_ikr,
            eps_ckg,
            timesteps
        )


        return eps_fused