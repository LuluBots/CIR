from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import AttenTokenPoolingLayer
from layers import StackedTransformer
# from scenic.projects.baselines.clip import layers as clip_layers
# from scenic.projects.baselines.clip import model as clip_model
import clip
import torchvision.transforms as transforms
import numpy as np
MagicLensConfig = {
    'base': dict(
        embed_dim=512,
        ff_hidden_size=512 * 4,
        num_layers=4,
        num_heads=8,
        num_query_token=1,
        # clip_model_name='vit_b16',
        clip_model_name='ViT-B/16',
    ),
    'large': dict(
        embed_dim=768,
        ff_hidden_size=768 * 4,
        num_layers=4,
        num_heads=16,
        num_query_token=1,
        # clip_model_name='vit_l14',
        clip_model_name='ViT-L/14',
    ),
}

def largest_square_crop(images: torch.Tensor) -> torch.Tensor:
    assert images.ndim >= 4
    _, _, h, w = images.shape
    size = min(h, w)

    pos_h = (h - w) // 2 if h > w else 0
    pos_w = (w - h) // 2 if w > h else 0

    return images[..., pos_h:pos_h + size, pos_w:pos_w + size]

class MagicLens(nn.Module):
    """MagicLens model built upon CLIP."""
    def __init__(self, model_size: str = 'base'):
        super().__init__()

        self.config = MagicLensConfig[model_size]
        self.clip = clip.load(self.config['clip_model_name'])[0]  
        self.size = 224  

        self.multimodal_encoder = StackedTransformer(
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            input_dim=self.config['embed_dim'],
            hidden_dim=self.config['ff_hidden_size'],
            use_bias=True,
            add_skip_connection=True,
            use_per_dim_scale=False,
        )

        self.contrastive_multimodal_pooler = AttenTokenPoolingLayer(
            input_dim=self.config['embed_dim'],
            query_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_query_tokens=self.config['num_query_token'],
            use_bias=True,
            use_per_dim_scale=True,
        )


    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Center crop & resize image to be compatible with the underlying vision model."""
        # print("images shape before _preprocess_images: ", images.shape)  # torch.Size([32, 224, 224, 3])
        assert images.ndim >= 4
        images = images.permute(0, 3, 1, 2)  # 转换为(B, C, H, W)
        images = largest_square_crop(images)  
        print("images shape 1: ", images.shape)  # torch.Size([32, 3, 224, 224])
        images = F.interpolate(images, size=(self.size, self.size), mode='bilinear', align_corners=False)
        print("images shape 2: ", images.shape)  # torch.Size([32, 3, 224, 224])
        return images



    def clip_encode(self, input_batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes CLIP embeds for the given batch of images and texts.
        
        Args:
        input_batch: A Dict of the following fields:
        * ids: [B, T] or [B, 1, T]. Text token ids
        * paddings: [B, T] or [B, 1, T]. Text token paddings.
        * image: [B, H, W, 3]. Input image.

        Returns:
        image_embs: [B, D]
        text_embs: [B, D]
        patch_embeds: [B, N, D]
        token_embds: [B, T, D]

        """
        assert input_batch['ids'].ndim <= 3
        print("input_batch['ids'] shape:", input_batch['ids'].shape) # shape: torch.Size([32, 1, 77])
        print("input_batch['image'] shape:", input_batch['image'].shape) # shape: torch.Size([32, 224, 224, 3])

        if input_batch['ids'].ndim == 3:
            # Only takes the first caption.
            input_batch['ids'] = input_batch['ids'][:, 0, :]
            input_batch['ids'] = input_batch['ids'].long()

        if isinstance(input_batch['ids'], np.ndarray):
            input_batch['ids'] = torch.tensor(input_batch['ids'], dtype=torch.long)

        images = self._preprocess_images(input_batch['image'])
        print("Processed image shape:", images.shape)  # torch.Size([B, 3, 224, 224])
        image_embs, text_embs = self.clip(images, input_batch['ids'])
        print("Image embeddings shape:", image_embs.shape)  # Ensure it matches [B, D]
        print("Text embeddings shape:", text_embs.shape)  # Ensure it matches [B, D]
        return image_embs, text_embs

    def _normalize_embed(self, embed: torch.Tensor) -> torch.Tensor:
        """Applies normalization on the input embedding.
        
        Args:
        embed: [B, D]. The input embedding to normalize.

        Returns:
        The normalized embedding.
        
        """
        # Always converts embed to float32 for all precisions.
        embed = embed.float()  
        # return py_utils.l2_normalize(embed, axis=-1)
        norm = torch.norm(embed, p=2, dim=-1, keepdim=True) + 1e-12
        return embed / norm

    def forward(self, input_batch: Dict) -> Dict:
        """Computes the multimodal embeddings.

        It computes the multimodal embeddings pooling from both
        text embeddings and image *generative* embeddings.
        If text is empty, use image pooling only.

        Args:
        input_batch: A Dict of the following fields:
            * ids: [B, T] or [B, 1, T]. Text token ids
            * paddings: [B, T] or [B, 1, T]. Text token paddings.
            * image: [B, H, W, 3]. Input image.
        Returns:
        A Dict contains the following fields:
            * multimodal_embed: [B, D], multimodal embedding
            * multimodal_embed_norm: [B, D], normalized multimodal embedding
        """

        img_embed, txt_embed = self.clip_encode(input_batch)  # [B, D], [B, D]
        print("1:",img_embed.shape,txt_embed.shape)  # torch.Size([32, 512]) torch.Size([32, 512])
        img_embed = img_embed.view(-1, 1, img_embed.size(-1))  # [B, 1, D]
        print("2:",img_embed.shape)  # torch.Size([32, 1, 512])
        txt_embed = txt_embed.view(-1, 1, txt_embed.size(-1))  # [B, 1, D]
        print("3:",txt_embed.shape)  # torch.Size([32, 1, 512])

        concate_mm_embed = torch.cat([img_embed, txt_embed], dim=1)
        print("4:",concate_mm_embed.shape) # torch.Size([32, 2, 512])
        multimodal_embed = self.multimodal_encoder(concate_mm_embed)  # [B, 2, D] torch.Size([20, 2, 512])
        print("5:",multimodal_embed.shape)
        multimodal_embed = self.contrastive_multimodal_pooler(multimodal_embed)
        multimodal_embed = multimodal_embed[:, 0]

        multimodal_embed_norm = self._normalize_embed(multimodal_embed)

        # placeholder for model matching
        # contrastive_loss = 0.0

        return {
            'multimodal_embed': multimodal_embed,
            'multimodal_embed_norm': multimodal_embed_norm,
        }

