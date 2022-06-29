import os
import json
import numpy as np
from PIL import Image

import torch
torch.no_grad()

# parameter load in utility
from .text_tokenizer import TextTokenizer
from .load_params import load_dalle_bart_flax_params, load_vqgan_torch_params, convert_dalle_bart_torch_from_flax_params

# models
from .models.vqgan_detokenizer import VQGanDetokenizer
from .models.dalle_bart_encoder_torch import DalleBartEncoderTorch
from .models.dalle_bart_decoder_torch import DalleBartDecoderTorch

"""
Utils
"""

def load_dalle_bart_metadata(path):
    print("parsing metadata from {}".format(path))
    for f in ['config.json', 'flax_model.msgpack', 'vocab.json', 'merges.txt']:
        assert(os.path.exists(os.path.join(path, f)))
    with open(path + '/config.json', 'r') as f: 
        config = json.load(f)
    with open(path + '/vocab.json') as f:
        vocab = json.load(f)
    with open(path + '/merges.txt') as f:
        merges = f.read().split("\n")[1:-1]
    return config, vocab, merges

"""
Class
"""

class DalleMini():
    """
    DalleMini

    Loads in models from disk, allows for inference using make_promptart
    """
    def __init__(self, is_mega=False):
        if torch.cuda.is_available(): print("Using CUDA!")

        # load dall e model
        model_name = 'mega' if is_mega else 'mini'
        print(f"Preparing dalle-{model_name}")

        model_path = f'./pretrained/dalle_bart_{model_name}'

        self.config, self.vocab, self.merges = load_dalle_bart_metadata(model_path)

        params_dalle_bart = load_dalle_bart_flax_params(model_path)

        # text tokens to image tokens
        self.encoder = self._init_encoder(params_dalle_bart)
        self.decoder = self._init_decoder(params_dalle_bart)

        # image tokens to image
        self.detokenizer = self._init_detokenizer()

    """
    The function
    """
    def make_promptart(self, prompt, seed=0):
        torch.manual_seed(seed)
        
        # text to tokens
        print("Encoding text to tokens...")
        text_tokens = self._tokenize_text(prompt)

        # text tokens to image tokens
        print("Translating text tokens to image tokens...")
        encoder_state = self.encoder(text_tokens)

        # only decoder is on cpu for now
        if torch.cuda.is_available(): 
            text_tokens = text_tokens.cuda()
            encoder_state = encoder_state.cuda()
            self.decoder = self.decoder.cuda()
            print("PUSHED TO GPU")

        image_tokens = self.decoder.forward(text_tokens, encoder_state)

        # image tokens to image
        print("Decoding image tokens to image...")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8).detach().cpu().numpy()

        return Image.fromarray(image)


    """
    Init Help
    """
    def _init_encoder(self, params_dalle_bart):
        encoder = DalleBartEncoderTorch(
            layer_count = self.config['encoder_layers'],
            embed_count = self.config['d_model'],
            attention_head_count = self.config['encoder_attention_heads'],
            text_vocab_count = self.config['encoder_vocab_size'],
            text_token_count = self.config['max_text_length'],
            glu_embed_count = self.config['encoder_ffn_dim']
        )
        encoder_params = convert_dalle_bart_torch_from_flax_params(
            params_dalle_bart.pop('encoder'), 
            layer_count=self.config['encoder_layers'], 
            is_encoder=True
        )

        encoder.load_state_dict(encoder_params, strict=False)
        del encoder_params

        return encoder

    def _init_decoder(self, params_dalle_bart):
        decoder = DalleBartDecoderTorch(
            image_vocab_size = self.config['image_vocab_size'],
            image_token_count = self.config['image_length'],
            sample_token_count = self.config['image_length'], # ?
            embed_count = self.config['d_model'],
            attention_head_count = self.config['decoder_attention_heads'],
            glu_embed_count = self.config['decoder_ffn_dim'],
            layer_count = self.config['decoder_layers'],
            batch_count = 2,
            start_token = self.config['decoder_start_token_id'],
            is_verbose = True
        )

        decoder_params = convert_dalle_bart_torch_from_flax_params(
            params_dalle_bart.pop('decoder'), 
            layer_count=self.config['decoder_layers'],
            is_encoder=False
        )

        decoder.load_state_dict(decoder_params, strict=False)
        del decoder_params

        return decoder

    def _init_detokenizer(self, model_path ='./pretrained/vqgan'):
        # load parameters from disk
        vqgan_parameters = load_vqgan_torch_params(model_path)

        # create vqgan
        detokenizer = VQGanDetokenizer()
        detokenizer.load_state_dict(vqgan_parameters)
        del vqgan_parameters

        return detokenizer

    """
    Inference Help
    """
    def _tokenize_text(self, text):
        tokens = TextTokenizer(self.vocab, self.merges)(text)

        text_tokens = np.ones((2, self.config['max_text_length']), dtype=np.int32)
        text_tokens[0, :len(tokens)] = tokens
        text_tokens[1, :2] = [tokens[0], tokens[-1]]

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)

        return text_tokens