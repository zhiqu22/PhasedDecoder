from typing import Any, Dict, List, Optional
from torch import Tensor
import torch
import torch.nn as nn
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import transformer_layer
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerDecoderPDECBase':
        return 'TransformerDecoderPDEC'
    else:
        return module_name
    
class TransformerDecoderPDECBase(TransformerDecoderBase):

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.decoder_langtok = cfg.decoder_langtok
        self.attention_position_bias = cfg.attention_position_bias
        self.adaption_flag = True if cfg.adaption_flag == "True" else False
        if self.adaption_flag:
            self.fc1_input = self.build_fc1(self.embed_dim, cfg.adaption_inner_size, cfg.quant_noise.pq, cfg.quant_noise.pq_block_size,)
            self.fc2_input = self.build_fc2(cfg.adaption_inner_size, self.embed_dim, cfg.quant_noise.pq, cfg.quant_noise.pq_block_size,)
            self.relu_input = nn.functional.relu
            self.dropout_module_input = FairseqDropout(cfg.adaption_dropout, module_name=self.__class__.__name__)
            self.fc1_output = self.build_fc1(self.embed_dim, cfg.adaption_inner_size, cfg.quant_noise.pq, cfg.quant_noise.pq_block_size,)
            self.fc2_output = self.build_fc2(cfg.adaption_inner_size, self.embed_dim, cfg.quant_noise.pq, cfg.quant_noise.pq_block_size,)
            self.relu_output = torch.nn.functional.relu
            self.dropout_module_output = FairseqDropout(cfg.adaption_dropout, module_name=self.__class__.__name__)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        # no cross-attention
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn=True)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            contrastive_flag: bool = False,
            contrastive_position = None,
            identity_flag = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            contrastive_flag=contrastive_flag,
            contrastive_position=contrastive_position,
            identity_flag = identity_flag,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            contrastive_flag: bool = False,
            contrastive_position = None,
            identity_flag = False,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            contrastive_flag,
            contrastive_position,
            identity_flag = identity_flag,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        contrastive_flag: bool = False,
        contrastive_position = None,
        identity_flag = False,
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if self.adaption_flag:
            input = self.dropout_module_input(self.fc2_input(self.relu_input(self.fc1_input(enc))))
            enc = enc + input
        
        # cat source and target
        cat_flag = False
        # incremental_state is None means it is in training.
        # (incremental_state is not None and incremental_state == {}) means this is the first turn of incremental decoding.
        if incremental_state is None or (incremental_state is not None and incremental_state == {}):
            x = torch.cat([enc, x], dim=0)
            cat_flag = True
            tokens = torch.cat([encoder_out["src_tokens"], prev_output_tokens], dim=1)

        self_attn_padding_mask: Optional[Tensor] = None
        if cat_flag:
            if tokens.eq(self.padding_idx).any():
                self_attn_padding_mask = tokens.eq(self.padding_idx)
        else:
            if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
                self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        
            
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        self_attn_mask = self.build_future_mask(x, enc.shape[0], mask_flag=(incremental_state is None or (incremental_state is not None and incremental_state == {})) )
        src_instructions = None
        tgt_instructions = None
        src_length = enc.shape[0]
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            if cat_flag:
                if contrastive_flag:
                    # break for saving cost
                    if identity_flag and contrastive_position < idx: break
                    tmp = x.transpose(0, 1)
                    if contrastive_position == idx:
                        if not self.decoder_langtok:
                            indices_tag = torch.argmax(tokens, dim=-1)
                            src_instructions = tmp[range(bs), indices_tag, :]
                        else:
                            max_vals, _ = torch.max(tokens, dim=1, keepdim=True)
                            coords = (tokens == max_vals).nonzero()
                            result_coords = coords[:, 1].reshape(bs, 2)
                            src_instructions = tmp[range(bs),result_coords[:,0],:]
                            tgt_instructions = tmp[range(bs),result_coords[:,1],:]
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
        
        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if cat_flag:
            x = x[src_length:,:,:]
        if self.adaption_flag:
            output = self.dropout_module_output(self.fc2_output(self.relu_output(self.fc1_output(x))))
            x = x + output

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # since the flow is concatenated, we have to remove the source part from flow.
        
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x, {"attn": [attn],
                   "inner_states": inner_states, 
                   "src_instructions":src_instructions,
                   "tgt_instructions":tgt_instructions}
    
    def build_future_mask(self, tensor, src_length, mask_flag):
        if mask_flag:
            # in training
            dim = tensor.size(0)
            if (
                self._future_mask.size(0) == 0
                or self._future_mask.size(0) < dim
                ):
                self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
            if self._future_mask.device == tensor.device:
                tmp = self._future_mask[:dim, :dim].clone()
            else:
                tmp = self._future_mask[:dim, :dim].to(tensor, copy=True)
            tmp[ :src_length + self.attention_position_bias, :src_length + self.attention_position_bias] = 0.
            return tmp
        else:
            return None
    
class TransformerDecoderPDEC(TransformerDecoderPDECBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )