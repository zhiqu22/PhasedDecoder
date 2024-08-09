from typing import Optional

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_legacy import (
    TransformerModel
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
)
from PhasedDecoder.models.transformer_decoder_pdec import TransformerDecoderPDEC
from PhasedDecoder.models.transformer_encoder_pdec import TransformerEncoderPDEC


@register_model("transformer_pdec")
class TransformerModelPDEC(TransformerModel):
    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
        if self.cfg.encoder_layers == 0:
            self.encoder.layer_norm = None

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        """Add model-specific arguments to the parser."""
        parser.add_argument('--attention-position-bias', default=0, type=int)
        parser.add_argument('--adaption-flag', default="False", type=str)
        parser.add_argument('--adaption-inner-size', default=2048, type=int)
        parser.add_argument('--adaption-dropout', default=0.2, type=float)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderPDEC(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderPDEC(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        if 'contrastive_flag' in kwargs:
            contrastive_flag = True
            contrastive_position = kwargs.get("contrastive_position")
            identity_flag = kwargs.get("identity_flag")
        else:
            contrastive_flag = False
            contrastive_position = None
            identity_flag = False

        # kwargs.get return tuple, but we need tensor, so add [0]
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            contrastive_flag=contrastive_flag,
            contrastive_position=contrastive_position,
            identity_flag=identity_flag,
        )
        return decoder_out


@register_model_architecture("transformer_pdec", "transformer_pdec")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.merge_src_tgt_embed = getattr(args, "merge_src_tgt_embed", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

@register_model_architecture("transformer_pdec", "transformer_pdec_6_e_6_d")
def transformer_pdec_6_e_6_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
@register_model_architecture("transformer_pdec", "transformer_pdec_0_e_12_d")
def transformer_pdec_0_e_12_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_0_e_12_d_iwslt")
def transformer_pdec_0_e_12_d_iwslt(args):
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)
    
@register_model_architecture("transformer_pdec", "transformer_pdec_1_e_11_d")
def transformer_pdec_1_e_11_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.decoder_layers = getattr(args, "decoder_layers", 11)
    base_architecture(args)
@register_model_architecture("transformer_pdec", "transformer_pdec_3_e_9_d")
def transformer_pdec_3_e_9_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_layers = getattr(args, "decoder_layers", 9)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_5_e_7_d")
def transformer_pdec_5_e_7_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.decoder_layers = getattr(args, "decoder_layers", 7)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_7_e_5_d")
def transformer_pdec_7_e_5_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 7)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_9_e_3_d")
def transformer_pdec_9_e_3_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 9)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)
    
@register_model_architecture("transformer_pdec", "transformer_pdec_11_e_1_d")
def transformer_pdec_11_e_1_d(args):
    args.encoder_layers = getattr(args, "encoder_layers", 11)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_big")
def transformer_pdec_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_big_only")
def transformer_pdec_big_only(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_big_1024")
def transformer_pdec_big_1024(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)

@register_model_architecture("transformer_pdec", "transformer_pdec_big_1024_only")
def transformer_pdec_big_1024_only(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    base_architecture(args)