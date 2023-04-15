class MedConfig:
    def __init__(self):
        self.architectures = ["BertModel"]
        self.attention_probs_dropout_prob = 0.1
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.initializer_range = 0.02
        self.intermediate_size = 3072
        self.layer_norm_eps = 1e-12
        self.max_position_embeddings = 512
        self.model_type = "bert"
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.pad_token_id = 0
        self.type_vocab_size = 2
        self.vocab_size = 30524
        self.encoder_width = 768
        self.add_cross_attention = True
