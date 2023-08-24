import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from VIT import * 

class BINMODEL(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        inner_encoder, 
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.InnerEncoder = inner_encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:] # [1,257,768], 257,768
        self.to_patch, _ = encoder.to_patch_embedding_[:2]
        self.patch_to_emb = encoder.to_patch_embedding  
        #self.to_patch - Rearange.. , self.patch_to_emb - Linear Layer
        #pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        pixel_values_per_patch = 256

        # decoder parameters

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, gt_img):
        device = img.device
        tokens = self.patch_to_emb(img) # Pass through Linear layer
        b, n, _ = tokens.shape
        tokens = tokens + self.encoder.pos_embedding[:, :(n)]
    

        encoded_tokens = self.encoder.transformer(tokens) # Without CLS Token
        #print("Encoded Tokens :",encoded_tokens.shape) # Main Patch

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        #print(decoder_tokens.shape)

        # attend with decoder

        decoded_tokens = self.decoder(decoder_tokens)

        # project to pixel values
        #print("Decoded Tokens Shape",decoded_tokens.shape)
        pred_pixel_values = self.to_pixels(decoded_tokens)
        
        # calculate reconstruction loss with gt
        
        gt_patches = self.to_patch(gt_img)
        #print("Pred Pixels Shape",pred_pixel_values.shape)
        #print("GT Image Shape",gt_patches.shape)
        recon_loss = F.mse_loss(pred_pixel_values, gt_patches)

        return recon_loss, pred_pixel_values
