import torch
import torch.nn as nn
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from biomedclip.vision_modules import Block
from biomedclip.text_modules import BertLayer
from typing import Optional
from .layers import PVL_Adapter
from .scale_block import ScaleBlock


def load_biomedclip_to_device(cfg):
    
    model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    vision_target_network = nn.Sequential(*[Block(768,12) for i in range(12)]).to(cfg.MODEL.DEVICE)
    vision_network = model.visual.trunk.blocks.to(cfg.MODEL.DEVICE)

    text_target_network = nn.ModuleList([BertLayer() for i in range(12)]).to(cfg.MODEL.DEVICE)
    text_network = model.text.transformer.encoder.layer.to(cfg.MODEL.DEVICE)

    for target_param, param in zip(vision_target_network.parameters(), vision_network.parameters()):
            target_param.data.copy_(param.data)

    for target_param, param in zip(text_target_network.parameters(), text_network.parameters()):
            target_param.data.copy_(param.data)

    model.visual.trunk.blocks = vision_target_network.to(cfg.MODEL.DEVICE)
    model.text.transformer.encoder.layer  = text_target_network.to(cfg.MODEL.DEVICE)

    return model.to(cfg.MODEL.DEVICE).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super(CustomCLIP, self).__init__()
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text
        self.logit_scale = clip_model.logit_scale

        self.fusion_stages = cfg.MODEL.LAYERS

        if(cfg.MODEL.BACKBONE == "ViT-B/16"):
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512
            
        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.temperature = cfg.MODEL.TEMPERATURE
        self.beta = cfg.MODEL.BETA
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
            PVL_Adapter(in_channels_vis=self.embed_dim, in_channels_txt=self.embed_dim, adapter_channels=adapter_channels, 
                            beta=self.beta, gate_init=self.gate_init)
            for _ in range(len(self.fusion_stages))
        ])

    def encode_text_image(self,  tokenized_prompts, text_prompts, image, 
                            attention_mask: Optional[torch.LongTensor] = None):

        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()
        
        x_txt = self.text_model.transformer.embeddings(
            inputs_embeds=text_prompts
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        trunk = self.vision_model.trunk
        x_img = trunk.patch_embed(image)
        B, HW, C = x_img.shape
        x_img = trunk._pos_embed(x_img)
        x_img = trunk.norm_pre(x_img)

        hidden_states = []

        for i, (block, layer) in enumerate(zip(trunk.blocks,self.text_model.transformer.encoder.layer)):
            if(i == 0):
                x_img = block(x_img)
                x_txt = layer(x_txt, attention_mask=extended_attention_mask)
            elif(i in self.fusion_stages):
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img, x_txt)

                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl

                x_img = block(x_img)
                x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            else:
                x_img = block(x_img) 
                x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            hidden_states.append(x_img)
            x_txt = x_txt[0]
        
        x_img = trunk.norm(x_img)
        # Linear Projection: 768 -> 512
        x_img = self.vision_model.head(x_img)

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)
        x_txt = self.text_model.proj(x_txt)

        if self.output_hidden_states:
            return x_img, hidden_states, projected
        else:
            return x_img, projected

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(1)
        cls_token = image_features[:, 0, :]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        seg_logits = image_features[:, 1:, :]
        seg_logits = seg_logits / seg_logits.norm(dim=-1, keepdim=True)

        seg_logits = seg_logits.reshape(B, H//self.patch_size, W//self.patch_size, -1).permute(0,3,1,2)
        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(text_features), self.upscale(seg_logits)
        )
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear", align_corners=False).squeeze(1)
        return seg_logits, cls_token
    
    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def forward(self, image, text, num_samples=30):

        B, C, H, W = image.shape
        
        tokenized_prompts = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)
        
        # regular forward
        image_features, text_features = self.encode_text_image(
            tokenized_prompts, prompts, image
        )

        seg_logits,cls_token = self.compute_seg_logits(image_features, text_features, 
                                                B,H,W)
        
        if(self.training):
        
        
            patch_logits = image_features[:, 1:, :]
            patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
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
                    tokenized_prompts, prompts, image
                )
                seg_logits, _ = self.compute_seg_logits(image_features, text_features, B, H, W)
                seg_samples.append(seg_logits)

            seg_samples = torch.stack(seg_samples, dim=0)  # [N, B, C, H, W]

            return seg_samples
        
def build_medclipseg_biomedclip(cfg):

    print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_biomedclip_to_device(cfg)

    clip_model.float()

    print("Building custom BiomedCLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():
        if "mask_head" in name:
            param.requires_grad_(True)
        elif "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model