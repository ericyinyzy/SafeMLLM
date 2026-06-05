import torch
from llava.conversation import conv_llava_v1,conv_llava_llama_2#,conv_llava_llama_2_soft
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from safetensors import safe_open
from torch import nn
def prepare_text_prompt(user_prompt,softprompt=None):
    # sys_prompt=''.join([f'<soft_prompt_{i}>' for i in range(116)])

    qs = DEFAULT_IMAGE_TOKEN + '\n' + user_prompt
    # qs = DEFAULT_IMAGE_TOKEN + '\n' + user_prompt
    conv = conv_llava_v1.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print('prompt',prompt)

    return prompt

# support batch implementation
class Prompt:
    # tokenization
    # turn to embeddings

    # padding? wait until targets have been appended
    # prepare labels? need to wait for targets

    def __init__(self, model, tokenizer, text_prompts=None, device='cuda:0'):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.text_prompts = text_prompts
        self.context_length = []
        self.input_ids = []
        self.do_tokenization(self.text_prompts)


    def do_tokenization(self, text_prompts):

        # if text_prompts is None:
        #     self.input_ids = []
        #     self.context_length = []
        #     return
        #
        # input_ids = tokenizer_image_token(text_prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #
        # self.input_ids = [input_ids]
        # self.context_length = [input_ids.shape[1]]
        self.input_ids = []
        self.context_length = []
        if text_prompts is None:
            self.input_ids = []
            self.context_length = []
            return
        else:
            for text_prompt in text_prompts:
                input_ids = tokenizer_image_token(text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                  return_tensors='pt').unsqueeze(0).cuda()
                # print(text_prompt,len(input_ids))
                # print()

                self.input_ids.append(input_ids)
                self.context_length.append(input_ids.shape[1])
        # exit()