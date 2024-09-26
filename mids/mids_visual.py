import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, CLIPVisionModel
from mids.basic_module import MultiHeadCrossAttention, MultiHeadSelfAttention

class MultiModalFusionBlock(nn.Module):
    def __init__(self, dim=768):
        super(MultiModalFusionBlock, self).__init__()

        self.cr_tii1 = MultiHeadCrossAttention(dim, 8)
        self.cr_tii2 = MultiHeadCrossAttention(dim, 8)
        self.cr_itt1 = MultiHeadCrossAttention(dim, 8)
        self.cr_itt2 = MultiHeadCrossAttention(dim, 8)

        self.lr_tii1 = nn.Parameter(torch.randn(1, dim))
        self.cr_tii1_lr = MultiHeadCrossAttention(dim, 8)

        self.lr_tii2 = nn.Parameter(torch.randn(1, dim))
        self.cr_tii2_lr = MultiHeadCrossAttention(dim, 8)

        self.lr_itt1 = nn.Parameter(torch.randn(1, dim))
        self.cr_itt1_lr = MultiHeadCrossAttention(dim, 8)

        self.lr_itt2 = nn.Parameter(torch.randn(1, dim))
        self.cr_itt2_lr = MultiHeadCrossAttention(dim, 8)
        
    
    def forward(self, layer1_rgb_feature, layer2_rgb_feature, texts_features):
        batch_size = texts_features.size(0)
        samples_num = texts_features.size(1)
        dim = texts_features.size(3)

        layer1_rgb_features = layer1_rgb_feature.unsqueeze(1).expand(batch_size, samples_num, -1, dim)
        layer1_rgb_features = layer1_rgb_features.reshape(batch_size*samples_num, -1, dim)
        layer2_rgb_features = layer2_rgb_feature.unsqueeze(1).expand(batch_size, samples_num, -1, dim)
        layer2_rgb_features = layer2_rgb_features.reshape(batch_size*samples_num, -1, dim)

        texts_features = texts_features.reshape(batch_size*samples_num, -1, dim)

        tii1 = self.cr_tii1(texts_features, layer1_rgb_features)
        itt1 = self.cr_itt1(layer1_rgb_features, texts_features)
        tii2 = self.cr_tii2(texts_features, layer2_rgb_features)
        itt2 = self.cr_itt2(layer2_rgb_features, texts_features)

        tii1_feature = self.cr_tii1_lr(self.lr_tii1.expand(batch_size*samples_num, 1, dim), tii1)
        itt1_feature = self.cr_itt1_lr(self.lr_itt1.expand(batch_size*samples_num, 1, dim), itt1)
        tii2_feature = self.cr_tii2_lr(self.lr_tii2.expand(batch_size*samples_num, 1, dim), tii2)
        itt2_feature = self.cr_itt2_lr(self.lr_itt2.expand(batch_size*samples_num, 1, dim), itt2)

        return (tii1_feature, itt1_feature, tii2_feature, itt2_feature)

"""
NOTE: This version is temporarily employed to visualize the heatmap with GradCAM.
Due to the author's limitied time, the integration with MIDS, 
which would simultaneously output text analysis and heatmaps, will be postponed. 
At that time, this version will be deprecated.
"""
class MIDSvisual(nn.Module):
    def __init__(self, dim=768, **kwargs):
        super(MIDSvisual, self).__init__()

        self.texts = kwargs.pop('texts', None)
        self.text_model_path = kwargs.pop('text_model_path', 'models/t5-base')
        self.image_model_path = kwargs.pop('image_model_path', 'models/clip-vit-large-patch14-336')

        # Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(self.text_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_path, use_fast=False, legacy=False)

        # Image rgb encoder
        self.image_encoder = CLIPVisionModel.from_pretrained(self.image_model_path)

        # Projector
        self.projector_layer1 = nn.Linear(1024, dim)
        self.projector_layer2 = nn.Linear(1024, dim)
        self.cls_token_layer1 = nn.Linear(dim, dim)
        self.cls_token_layer2 = nn.Linear(dim, dim)

        # multimodal fusion
        self.fusion_block = MultiModalFusionBlock(dim)
        self.se_attn = nn.ModuleList([MultiHeadSelfAttention(dim, 8) for _ in range(3)])

        # classifier
        self.classifier = nn.Linear(dim*6, 4)
    
    def _encode_text(self, inputs):
        text_outputs = self.text_encoder(**inputs)
        text_features = text_outputs.last_hidden_state
        return text_features

    def _encode_image(self, inputs):
        image_features = self.image_encoder(inputs, output_hidden_states=True)
        layer1_rgb_feature = image_features.hidden_states[-1]
        layer2_rgb_feature = image_features.hidden_states[-2]
        return layer1_rgb_feature, layer2_rgb_feature

    def forward(self, image, batch_size=1, N=0, M=0):
        input_ids = self.tokenizer(self.texts, return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True)
        texts_features = self._encode_text(input_ids) # (batch*(1+N+M), seq_len, dim)
        dim = texts_features.size(2)

        texts_features = texts_features.reshape(batch_size, 1+N+M, -1, dim)

        layer1_rgb_feature, layer2_rgb_feature = self._encode_image(image) # (batch, 577, 1024)
        layer1_rgb_feature = self.projector_layer1(layer1_rgb_feature) # (batch, 577, 768)
        layer2_rgb_feature = self.projector_layer2(layer2_rgb_feature) # (batch, 577, 768)

        cls_token1 = self.cls_token_layer1(layer1_rgb_feature[:,0:1,:])
        cls_token2 = self.cls_token_layer2(layer2_rgb_feature[:,0:1,:])
        cls_token1 = cls_token1.unsqueeze(1).expand(batch_size, 1+N+M, -1, dim).reshape(batch_size*(1+N+M), -1, dim)
        cls_token2 = cls_token2.unsqueeze(1).expand(batch_size, 1+N+M, -1, dim).reshape(batch_size*(1+N+M), -1, dim)

        # fusion
        tii1_feature, itt1_feature, tii2_feature, itt2_feature = self.fusion_block(layer1_rgb_feature, layer2_rgb_feature, texts_features)
        
        fused_features = torch.cat((cls_token1, tii1_feature, itt1_feature, cls_token2, tii2_feature, itt2_feature), dim=1)
        # self attention
        for layer in self.se_attn:
            fused_features = layer(fused_features)
        
        # answer classification
        logits = self.classifier(torch.flatten(fused_features, start_dim=1)) # (batch*sample, 4)
        
        logits = logits.reshape(batch_size, -1, 4) # (batch, sample, 4)

        return logits

    def freeze_text_encoder_parameters(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def freeze_image_encoder_parameters(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def unfreeze_vision_encoder(self, last_layers=2):
        for param in self.image_encoder.vision_model.encoder.layers[-last_layers:].parameters():
            param.requires_grad = True