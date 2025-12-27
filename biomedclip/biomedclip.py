import torch
import torch.nn as nn
from typing import Optional
from open_clip.hf_model import ClsPooler

class BiomedCLIP(nn.Module):
    def __init__(self, model, prompt_depth = None, prompt_length = None, output_hidden_states=False):
        super(BiomedCLIP, self).__init__()
        self.vision_model = model.visual
        self.text_model = model.text
        self.logit_scale = model.logit_scale
        self.prompt_depth = prompt_depth
        self.output_hidden_states = output_hidden_states
        self.prompt_tokens = nn.Parameter(torch.empty(prompt_depth, prompt_length, 768))
        self.text_dtype = self.text_model.transformer.dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        nn.init.normal_(self.prompt_tokens, mean=0, std=0.02)
    def encode_image(self, x):
        trunk = self.vision_model.trunk
        x = trunk.patch_embed(x)
        x = trunk._pos_embed(x)
        x = trunk.norm_pre(x)

        hidden_states = []

        for i, block in enumerate(trunk.blocks):
            if(i < self.prompt_depth - 1):
                x = block(x, self.prompt_tokens[i])
            else:
                x = block(x)

            hidden_states.append(x)

        x = trunk.norm(x)

        # if trunk.global_pool:
        #     x = (
        #         x[:, trunk.num_prefix_tokens :].mean(dim=1)
        #         if trunk.global_pool == "avg"
        #         else x[:, 0]
        #     )

        x = trunk.fc_norm(x)
        x = trunk.head(x)

        # Linear Projection: 768 -> 512
        x = self.vision_model.head(x)

        if self.output_hidden_states:
            return x, hidden_states
        else:
            return x
    def encode_text(self, x, attention_mask: Optional[torch.LongTensor] = None, output_hidden_states: bool = False,):

        if attention_mask is None:
            attention_mask = (x != self.text_model.config.pad_token_id).long()

        inputs_embeds = self.text_model.transformer.embeddings.word_embeddings(x).type(self.text_dtype).to(self.device)

        x = self.text_model.transformer.embeddings(
            inputs_embeds=inputs_embeds
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.text_dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.text_dtype).min

        for i, layer in enumerate(self.text_model.transformer.encoder.layer):
            if(i < self.prompt_depth - 1):
                x = layer(x, attention_mask=extended_attention_mask, prompt_tokens=self.prompt_tokens[i])
            else:
                x = layer(x, attention_mask=extended_attention_mask)
            x = x[0]

        pooled_out = x[:, 0, :]
        projected = self.text_model.proj(pooled_out)


        return projected
    