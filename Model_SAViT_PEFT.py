# savit_spa_mixpool_lrmha.py
from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)


def cosine_schedule(val_min: float, val_max: float, t: int, T: int) -> float:
    if T <= 1:
        return val_max
    cos_t = 0.5 * (1 + math.cos(math.pi * t / (T - 1)))
    # map cos_t in [0..1] to [val_max..val_min] (increasing with t)
    return val_min * cos_t + val_max * (1 - cos_t)

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=1.0,
                 lora_dropout=0.0, bias=True, freeze_base=True):
        super().__init__()
        self.r = int(r)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        if self.r > 0:
            self.A = nn.Linear(in_features, self.r, bias=False)
            self.B = nn.Linear(self.r, out_features, bias=False)
            # init per LoRA: A ~ N(0, 1/r), B = 0 so update starts at 0
            nn.init.normal_(self.A.weight, std=1.0 / self.r)
            nn.init.zeros_(self.B.weight)
            self.scaling = lora_alpha / self.r
            self.drop = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 0.0
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.B(self.A(self.drop(x))) * self.scaling
        return y


class LoRAAdapter(nn.Module):
    def __init__(self, dim, r=8, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        self.r = int(r)
        if self.r > 0:
            self.A = nn.Linear(dim, self.r, bias=False)
            self.B = nn.Linear(self.r, dim, bias=False)
            nn.init.normal_(self.A.weight, std=1.0 / self.r)
            nn.init.zeros_(self.B.weight)
            self.scaling = lora_alpha / self.r
            self.drop = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 0.0
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return torch.zeros_like(x)
        return self.B(self.A(self.drop(x))) * self.scaling


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)                    # [B, D, H', W']
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)    # [B, H'*W', D]
        return x, (Hp, Wp)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, freeze_base=True):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

        if freeze_base:
            for p in self.fc1.parameters():
                p.requires_grad = False
            for p in self.fc2.parameters():
                p.requires_grad = False

    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        # add trainable adaptation on top of frozen MLP output
        return y


class LrMSA(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                 qkv_bias=True, lora_rank=8, lora_alpha=1.0, lora_dropout=0.0,
                 freeze_base=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = LoRALinear(dim, dim, r=lora_rank, lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout, bias=qkv_bias,
                                 freeze_base=freeze_base)
        self.k_proj = LoRALinear(dim, dim, r=lora_rank, lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout, bias=qkv_bias,
                                 freeze_base=freeze_base)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if freeze_base:
            for p in self.qkv.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_delta = self.q_proj(x).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k_delta = self.k_proj(x).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        q = q + q_delta
        k = k + k_delta

        attn = (q @ k.transpose(-2, -1)) * self.scale                                         # [B,h,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D) # [B,N,D]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class LrViTBlock(nn.Module):
    def __init__(self, dim, num_heads, tau=4, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 lora_rank=8, lora_alpha=1.0, lora_dropout=0.0, freeze_base=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LrMSA(dim, num_heads, attn_drop, drop,
                          qkv_bias=True, lora_rank=lora_rank,
                          lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          freeze_base=freeze_base)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop, freeze_base=freeze_base)  # see MLP below

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        #self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Z: torch.Tensor, AP: torch.Tensor) -> torch.Tensor:
        B, Lz, D = Z.shape
        Lp = AP.shape[1]

        q = Z.view(B, Lz, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,Lz,d]
        k = self.k_proj(AP).view(B, Lp, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,Lp,d]
        v = self.v_proj(AP).view(B, Lp, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,Lp,d]

        attn = (q @ k.transpose(-2, -1)) * self.scale                                   # [B,h,Lz,Lp]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Lz, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SPA(nn.Module):
    def __init__(self, dim, num_heads, tau=4, depth=2, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 prompt_length: int = 0, use_cross_attn=True,
                 lora_rank=8, lora_alpha=1.0, lora_dropout=0.0, freeze_base=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(dim, num_heads, attn_drop, drop,
                                             lora_rank=lora_rank,
                                             lora_alpha=lora_alpha,
                                             lora_dropout=lora_dropout,
                                             freeze_base=freeze_base)
        self.blocks = nn.ModuleList([
            LrViTBlock(dim, num_heads, tau, mlp_ratio, drop, attn_drop, drop_path,
                       lora_rank=lora_rank, lora_alpha=lora_alpha,
                       lora_dropout=lora_dropout, freeze_base=freeze_base)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        regions: List[torch.Tensor],
        mu_min: float = 0.5,
        mu_max: float = 0.9,
    ) -> List[torch.Tensor]:

        K2 = len(regions)
        B, S2, D = regions[0].shape
        tilde = []
        AP_prev = None  # [B, S^2, D]

        for i in range(K2):
            Zi = regions[i]  # [B, S^2, D]
            #print(Zi.shape,K2) 
            if i == 0:
                Zi_ = Zi
            else:
                mu = cosine_schedule(mu_min, mu_max, i-1, K2-1)
                AP_i = mu * AP_prev + (1.0 - mu) * tilde[-1]   
                if self.use_cross_attn:
                    Zi_ = self.cross_attn(Zi, AP_i)
                else:
                    Zi_ = (Zi+AP_i)/2
                AP_prev = AP_i

            x = Zi_
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            tilde.append(x)
            if i == 0:
                AP_prev = x.detach()  
        return tilde

class MixPool(nn.Module):
    def __init__(self, dim, num_heads, tau=4, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 scales: Optional[List[Tuple[int, int, int, int]]] = None,
                 lora_rank=8, lora_alpha=1.0, lora_dropout=0.0, freeze_base=True):
        super().__init__()
        self.theta_mp = LrViTBlock(dim, num_heads, tau, mlp_ratio, drop, attn_drop, drop_path,
                                   lora_rank=lora_rank, lora_alpha=lora_alpha,
                                   lora_dropout=lora_dropout, freeze_base=freeze_base)
        self.norm = nn.LayerNorm(dim)
        self.scales = scales

    def forward(self, tilde_list: List[torch.Tensor], K: int) -> torch.Tensor:
        K2 = len(tilde_list)
        B, S2, D = tilde_list[0].shape
        S = int(math.sqrt(S2))
        assert S * S == S2, "S^2 must be perfect square (tokens per region)"

        # Build Z_mix: [B, S^2, K, K, D]
        Z_mix = torch.stack(tilde_list, dim=2)  # [B, S^2, K^2, D]
        Z_mix = Z_mix.view(B, S2, K, K, D)

        # For each token position t in [0..S^2-1], we pool across the KxK region grid
        # to get an [S x S] map per scale, then flatten -> [S^2, D].
        # We implement this efficiently by permuting dims to [B*S^2, D, K, K].
        Z_rg = Z_mix.permute(0, 1, 4, 2, 3).contiguous().view(B * S2, D, K, K)  # [B*S2, D, K, K]

        pooled_sum = torch.zeros(B * S2, D, S, S, device=Z_rg.device, dtype=Z_rg.dtype)

        if self.scales is None:
            p1 = F.adaptive_avg_pool2d(Z_rg, (S, S))
            p2 = F.avg_pool2d(Z_rg, kernel_size=3, stride=1, padding=1)
            p2 = F.adaptive_avg_pool2d(p2, (S, S))
            p3 = F.avg_pool2d(Z_rg, kernel_size=5, stride=1, padding=2)
            p3 = F.adaptive_avg_pool2d(p3, (S, S))
            pooled_sum = p1 + p2 + p3
        else:
            for (kh, kw, sh, sw) in self.scales:
                out = F.avg_pool2d(Z_rg, kernel_size=(kh, kw), stride=(sh, sw), ceil_mode=False)
                if out.shape[-2:] != (S, S):
                    out = F.adaptive_avg_pool2d(out, (S, S))
                pooled_sum = pooled_sum + out

        pooled_sum = pooled_sum.view(B, S2, D, S, S).mean(dim=(-1, -2))  # average spatial SxS (per spec eq. keeps S^2 length)
        x = self.theta_mp(pooled_sum)  
        x = self.norm(x)
        Z_out = x.mean(dim=1)           
        return Z_out

class SAViT(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_heads=12,
                 K=2,
                 spa_depth=2,
                 tau=4,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 use_cross_attn=False,
                 num_classes=0,
                 global_pool=False,
                 mix_scales: Optional[List[Tuple[int, int, int, int]]] = None,
                 add_cls_token=True,
                 # NEW: LoRA & freezing knobs
                 lora_rank=8, lora_alpha=1.0, lora_dropout=0.0, freeze_base=True):
        super().__init__()

        self.global_pool = global_pool
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        Hp, Wp = self.patch_embed.grid_size
        self.embed_dim = embed_dim
        self.add_cls_token = add_cls_token
        self.num_spatial = Hp * Wp

        pos_tokens = self.num_spatial + (1 if add_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_tokens, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)
            if freeze_base:
                self.cls_token.requires_grad = False
        else:
            self.register_parameter('cls_token', None)

        self.drop_after_pos = nn.Dropout(drop_rate)

        if freeze_base:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            self.pos_embed.requires_grad = False

        self.K = K
        assert Hp % K == 0 and Wp % K == 0, "Patch grid must be divisible by K"
        self.h_local = Hp // K
        self.w_local = Wp // K
        self.S2 = self.h_local * self.w_local
        self.K2 = K * K

        self.spa = SPA(
            dim=embed_dim,
            num_heads=num_heads,
            tau=tau,
            depth=spa_depth,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            prompt_length=self.S2,
            use_cross_attn=use_cross_attn,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, freeze_base=freeze_base,
        )

        self.mixpool = MixPool(
            dim=embed_dim,
            num_heads=num_heads,
            tau=tau,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            scales=mix_scales,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, freeze_base=freeze_base,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _geo_partition(self, I: torch.Tensor) -> List[torch.Tensor]:
        B, N, D = I.shape
        H = self.h_local * self.K
        W = self.w_local * self.K
        assert N == H * W, f"Token count mismatch: N={N}, H*W={H*W}"
        x_hw = I.view(B, H, W, D)
        parts = []
        for i in range(self.K):
            for j in range(self.K):
                patch = x_hw[:, i*self.h_local:(i+1)*self.h_local,
                             j*self.w_local:(j+1)*self.w_local, :] 
                parts.append(patch.reshape(B, self.S2, D))
        return parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens, (Hp, Wp) = self.patch_embed(x)
        if self.add_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        tokens = self.drop_after_pos(tokens)
        I = tokens[:, 1:, :] if self.add_cls_token else tokens 
        regions = self._geo_partition(I) 
        tilde_list = self.spa(regions)                        
        Z_out = self.mixpool(tilde_list, self.K)  
        if self.global_pool:
            Z_out = self.norm(Z_out)  # B L C
            #print(Z_out.shape)
        Z_out = self.head(Z_out)
        return Z_out

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

def SAViT_small_patch16(**kwargs):
    model = SAViT(
        img_size=224, patch_size=16, in_chans=3, embed_dim=384, num_heads=6, 
        K=2, spa_depth=10, tau=4, use_cross_attn=True, **kwargs
    )
    return model

def SAViT_base_patch16(**kwargs):
    model = SAViT(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_heads=12,
        K=2, spa_depth=11, tau=4, use_cross_attn=False, **kwargs
    )
    return model
