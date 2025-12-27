import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import PVL_Adapter
from .scale_block import ScaleBlock

from clip import clip

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MedCLIPSeg',
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, layer in enumerate(self.transformer.resblocks):
            x = self.transformer([x])[0]
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super(CustomCLIP, self).__init__()

        # === Core CLIP Components ===
        self.vision_model = clip_model.visual
        self.text_model = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.clip_model = clip_model

        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.text_emb_dim = 512
            self.patch_size = 16
            self.text_proj_dim = 512

        else:
            raise NotImplementedError("Other backbones not implemented yet.")

        self.temperature = cfg.MODEL.TEMPERATURE

        self.dtype = clip_model.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.beta = cfg.MODEL.BETA
        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE

        self.gate_init = cfg.MODEL.GATE_INIT
        self.cfg = cfg

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(in_channels_vis=self.embed_dim, in_channels_txt=self.text_emb_dim, adapter_channels=adapter_channels, 
                            beta=self.beta, gate_init=self.gate_init)
            for _ in range(len(self.fusion_stages))
        ])

    def encode_text_image(self,  tokenized_prompts, prompts, image):

        x_txt = prompts + self.text_model.positional_embedding.type(self.dtype)
        x_txt = x_txt.permute(1, 0, 2)  # NLD -> LND

        x_img = self.vision_model.conv1(image)  # shape = [*, width, grid, grid]
        B, C, H, W = x_img.shape
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_img = x_img.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_img = torch.cat(
            [self.vision_model.class_embedding.to(x_img.dtype) + torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
             x_img], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)

        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # NLD -> LND

        hidden_states = []

        for i, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks,self.text_model.transformer.resblocks)):

            if(i in self.fusion_stages):
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img.transpose(1,0), x_txt.transpose(1,0))
                x_img = x_img + vis_pvl.transpose(1,0)
                x_txt = x_txt + txt_pvl.transpose(1,0)

            x_img, hidden_states = block([x_img, hidden_states])
            x_txt = layer([x_txt])

            x_txt = x_txt[0]
        
        x_txt = x_txt.permute(1, 0, 2)  # LND -> NLD
        x_txt = self.text_model.ln_final(x_txt).type(self.dtype)
        x_txt = x_txt[torch.arange(x_txt.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_model.text_projection

        hidden_states = torch.stack(hidden_states, dim=0) # (Num Layers, L, N, D)
        x_patch =  hidden_states[:, 1:hidden_states.shape[1], :, :] # Remove visual ctx and class token
        x_patch = x_patch.permute(0, 2, 1, 3)  # LND -> NLD

        x_patch = x_patch[-1]

        x_cls =  hidden_states[-1, 0, :, :] # class token
        
        x_patch = self.vision_model.ln_post(x_patch)
        x_cls = self.vision_model.ln_post(x_cls)
        x_patch = x_patch @ self.vision_model.proj

        return x_patch, x_txt

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        seg_feats = image_features / image_features.norm(dim=-1, keepdim=True)

        h_patch = H // self.patch_size
        w_patch = W // self.patch_size
        seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2)

        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(text_features).unsqueeze(1), self.upscale(seg_feats)
        )
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear", align_corners=False).squeeze(1)
        return seg_logits

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def forward(self, image, text, num_samples=30):
        B, C, H, W = image.shape
        tokenized_prompts = torch.cat([clip.tokenize(t) for t in text]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            text_prompts = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        image_features, text_features = self.encode_text_image(
            tokenized_prompts, text_prompts, image
        )

        seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)

        if(self.training):
                patch_logits = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_mean = patch_logits.mean(dim=1)  # shape: (B, D)

                # Compute logits
                logits_per_image = (patch_mean @ text_features.T) / self.temperature   # (B, B)
                logits_per_text = (text_features @ patch_mean.T) / self.temperature  # (B, B)

                # --- Soft targets based on text similarity ---
                with torch.no_grad():
                    text_sim = (text_features @ text_features.T) / self.temperature # (B, B)
                    text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                    soft_targets = F.softmax(text_sim, dim=-1)  # temperature-controlled soft labels

                loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
                loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)

                clip_loss = (loss_i2t + loss_t2i) / 2

                return seg_logits, clip_loss
            
        else:

            seg_samples = []
            for _ in range(num_samples):

                image_features, text_features = self.encode_text_image(
                    tokenized_prompts, text_prompts, image
                )

                seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)
        
                seg_samples.append(seg_logits)

            seg_samples = torch.stack(seg_samples, dim=0)  # [N, B, C, H, W]

            return seg_samples 

def build_medclipseg_clip(cfg):

    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)

    return model