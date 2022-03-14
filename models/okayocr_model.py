from ast import arg
from email.policy import default
import torch.nn as nn
from fairseq.models import FairseqEncoder, register_model, FairseqEncoderDecoderModel, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, Embedding, TransformerModel
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import utils
from fairseq.models.transformer import base_architecture as base_transformer
from .swin.swin_transformer import SwinTransformer
from .unilm_models import UniLMDecoder
# from timm.models.vision_transformer import HybridEmbed, PatchEmbed, Block
from timm.models.layers import trunc_normal_
from timm.models import create_model
import torch
from torch.hub import load_state_dict_from_url
import os
from functools import partial
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model('OKayOCR_swin')
class OKayOCR_swin(FairseqEncoderDecoderModel):
    
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        # parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension')
        parser.add_argument(
            '--embed-dim', type=int, metavar='N',
            help='the patch size of h and w (h=w) of the ViT'
        )
        parser.add_argument(
            '--depths', type=list, metavar='N',
            help='the hidden size of the ViT'
        )
        parser.add_argument(
            '--num-heads', type=list, metavar='N',
            help='the layer num of the ViT'
        )
        parser.add_argument(
            '--reset-dictionary', action='store_true',
            help='if reset dictionary and related parameters'
        )
        parser.add_argument(
            '--adapt-dictionary', action='store_true',
            help='if adapt dictionary and related parameters'
        )
    @classmethod
    def build_model(cls, args, task):
        encoder = SwinEncoder(
            args = args,
            dictionary = task.source_dictionary
        )


        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        if getattr(args, "decoder_pretrained", None) == 'minilm':
            logger.info('Decoder is pretrained using the minilm.')
            
            prefix_of_parameter = 'bert'

            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )

            decoder = UniLMDecoder(
                args,
                task.target_dictionary,
                decoder_embed_tokens,
                no_encoder_attn=False,
            )            
            # load pretrained model
            if hasattr(args, 'decoder_pretrained_url') and args.decoder_pretrained_url != None and args.decoder_pretrained_url != '':                
                unilm_url = args.decoder_pretrained_url
                logger.info('The unilm model url: {}.'.format(unilm_url[:unilm_url.find('?')]))
                if unilm_url.startswith("http"):
                    unilm_state_dict = torch.hub.load_state_dict_from_url(unilm_url)            
                else:
                    unilm_state_dict = torch.load(unilm_url)
                unilm_layers = OrderedDict([(k, unilm_state_dict[k]) for k in unilm_state_dict.keys() if k.startswith(prefix_of_parameter + '.encoder.layer.')])
                unilm_layers_num = []
                for k in unilm_layers.keys():
                    t = k.replace(prefix_of_parameter + '.encoder.layer.', '')
                    t = t[:t.find('.')]
                    unilm_layers_num.append(int(t))
                unilm_layers_num = max(unilm_layers_num) + 1

                offset = unilm_layers_num - len(decoder.layers)
                assert offset == 0

                decoder_dict = decoder.state_dict()
                # embedding
                new_pos_weight = torch.zeros_like(decoder_dict['embed_positions.weight'])
                # position padding will right offset padding idx + 1
                new_pos_weight[task.target_dictionary.pad() + 1:, :] = unilm_state_dict[prefix_of_parameter + '.embeddings.position_embeddings.weight']
                new_decoder_dict = {
                    'embed_tokens.weight': unilm_state_dict[prefix_of_parameter + '.embeddings.word_embeddings.weight'],
                    'embed_positions.weight': new_pos_weight,
                    'layernorm_embedding.weight': unilm_state_dict[prefix_of_parameter + '.embeddings.LayerNorm.weight'],
                    'layernorm_embedding.bias': unilm_state_dict[prefix_of_parameter + '.embeddings.LayerNorm.bias']
                }            

                # layers
                key_map = {
                    'self_attn.k_proj': 'attention.self.key',
                    'self_attn.v_proj': 'attention.self.value',                
                    'self_attn.q_proj': 'attention.self.query',
                    'self_attn.out_proj': 'attention.output.dense',
                    'self_attn_layer_norm': 'attention.output.LayerNorm',
                    'fc1': 'intermediate.dense',
                    'fc2': 'output.dense',
                    'final_layer_norm': 'output.LayerNorm'
                }
                for layer_id in range(unilm_layers_num):
                    unilm_prefix = prefix_of_parameter + '.encoder.layer.{}.'.format(layer_id)
                    decoder_prefix = 'layers.{}.'.format(layer_id)

                    for key in key_map:
                        for suffix in ['.weight', '.bias']:
                            decoder_key = decoder_prefix + key + suffix
                            unilm_key = unilm_prefix + key_map[key] + suffix
                            if decoder_key in decoder_dict and unilm_key in unilm_state_dict:
                                new_decoder_dict[decoder_key] = unilm_state_dict[unilm_key]
                            
                if hasattr(args, "reset_dictionary") and args.reset_dictionary:
                    logger.info('Reset token embedding weights during decoder initialization.')
                    del new_decoder_dict['embed_tokens.weight']
                elif hasattr(args, "adapt_dictionary") and args.adapt_dictionary:
                    unilm_embed_tokens_weight = new_decoder_dict['embed_tokens.weight']
                    logger.info('Adapt token embedding weights during decoder initialization from {} to {}'.format(unilm_embed_tokens_weight.shape[0], decoder_embed_tokens.weight.shape[0]))                
                    new_decoder_dict['embed_tokens.weight'] = torch.zeros_like(decoder_dict['embed_tokens.weight'])
                    new_decoder_dict['embed_tokens.weight'][:min(unilm_embed_tokens_weight.shape[0], decoder_dict['embed_tokens.weight'].shape[0]), :] = unilm_embed_tokens_weight[:min(unilm_embed_tokens_weight.shape[0], decoder_dict['embed_tokens.weight'].shape[0]), :]

                missing_keys, unexpected_keys = decoder.load_state_dict(
                    new_decoder_dict, strict=False
                )

            else:
                logger.warning('You must specify the unilm model url or the decoder is randomly initialized.')

            # freeze k_proj bias
            for layer in decoder.layers:
                layer.self_attn.k_proj.bias.requires_grad = False
        else:
            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )

            decoder = TransformerDecoder(
                args = args,
                dictionary=task.target_dictionary,
                embed_tokens=decoder_embed_tokens,
                no_encoder_attn=False
            )

        model = cls(encoder, decoder)
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, imgs, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(imgs, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

@register_model_architecture('OKayOCR_swin', 'swin_tiny_patch4_window7')
def swin_tiny_patch4_window7(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.decoder_learned_pos = True
    args.layernorm_embedding = True
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 384)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.swin_arch = getattr(args, "swin_arch", "swin_tiny_patch4_window7")
    args.max_target_positions = 748
    base_transformer(args)

    

@register_model_architecture('OKayOCR_swin', 'swin_small_patch4_window7')
def swin_small_patch4_window7(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.decoder_learned_pos = True
    args.layernorm_embedding = True
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.swin_arch = getattr(args, "swin_arch", "swin_tiny_patch4_window7")
    args.max_target_positions = 512
    base_transformer(args)

class SwinEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.swin_transformer = create_model(args.swin_arch, args=args, pretrained=True)
        
        self.fp16 = args.fp16

    def forward(self, imgs):
        #if self.fp16:
        #    imgs = imgs.half()

        x, encoder_embedding = self.swin_transformer.forward_features(imgs)  # bs, n + 2, dim

        x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          _encoder_out = encoder_out['encoder_out'][0]
          _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
          _encoder_embedding = encoder_out['encoder_embedding'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [], 
                "src_tokens": [],
                "src_lengths": [],
        }

