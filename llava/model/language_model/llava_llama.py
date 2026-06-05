#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from typing import List, Optional, Tuple, Union
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config,tokenizer=None):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)
        self.tokenizer=tokenizer
        # print('config',config)
        # print(self.model.config)
        # exit()

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # print('print_model')
        # self.interval=self.model.config.interval
        self.input_ids=None
        self.attention_mask=None
        self.labels=None
        self.pad_token_id=0

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    #images=image_tensor.unsqueeze(0).half().cuda(),
                # do_sample=True,
                # temperature=args.temperature,
                # top_p=args.top_p,
                # num_beams=args.num_beams,
                # # no_repeat_ngram_size=3,
                # max_new_tokens=1024,
                # use_cache=True)
    def pgdattack(
        self,
        images,
        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # attn_loss: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print('input2',input_ids.shape)
        # images = images.clone().detach().to(torch.float).requires_grad_(True)
        shift_logits,shift_labels = self.prepare_inputs_labels_for_multimodal_attack(self.input_ids, self.attention_mask, None, self.labels, images,self.model,self.lm_head)
        return shift_logits,shift_labels
    def preparing_embedding_attack(self, batch_messages,batch_labels,soft_prompt):
        ### prepare input tokens
        # print('batch_messages',batch_messages,batch_labels)
        soft_prompt_front = soft_prompt[:int(soft_prompt.size(0) / 2)]
        soft_prompt_back = soft_prompt[int(soft_prompt.size(0) / 2):]
        n_prompt_tokens_front = soft_prompt_front.size(0)
        n_prompt_tokens_back = soft_prompt_back.size(0)
        n_prompt_tokens_total_front = n_prompt_tokens_front
        front_insert_list = [2 for _ in range(n_prompt_tokens_front)]
        back_insert_list = [2 for _ in range(n_prompt_tokens_back)]
        messages_with_eos_placeholder_ids = []

        for idx, messages in enumerate(batch_messages):
            for message in messages:
                new_message = message[:4] + front_insert_list + message[4:-4] + back_insert_list + message[-4:]
                messages_with_eos_placeholder_ids.append([new_message])
        messages_with_labels_ids = [
            label for label in
            batch_labels]

        input_lengths = []
        mask_length = []
        for e, et in zip(messages_with_eos_placeholder_ids, messages_with_labels_ids):
            input_lengths.append(len(e[0]) + len(et[0]))
            mask_length.append(len(e[0]))
        max_input_length = max(input_lengths)
        placeholder_start_index = 4
        placeholder_end_index = [len(messages_with_eos_placeholder_ids[i][0]) - 4 for i in
                                 range(len(messages_with_eos_placeholder_ids))]
        input_embeds_list = []
        label_list = []
        for idx, (e, et) in enumerate(zip(messages_with_eos_placeholder_ids, messages_with_labels_ids)):
            # print('reallyidx',idx)
            if len(e) == 1:
                # print('shizhema',idx)
                input_id = e[0] + et[0] + [self.pad_token_id] * (max_input_length - len(e[0]) - len(et[0]))
                to_regress_id = e[0] + et[0] + [-100] * (max_input_length - len(e[0]) - len(et[0]))

                to_regress_token_ids = torch.tensor(copy.deepcopy(to_regress_id),
                                                    dtype=torch.long).cuda()  # .to(model.device)#torch.ones([len(input_lengths), max_input_length],
                to_regress_token_ids[:mask_length[idx]] = -100
                label_list.append(to_regress_token_ids)
                # print('label',to_regress_token_ids,input_id)

                input_ids0 = torch.tensor(input_id, dtype=torch.long).cuda()  # .to(model.device)
                inputs_embeds = self.get_model().embed_tokens(input_ids0)
                inputs_embeds.requires_grad_(False)
                inputs_embeds[
                placeholder_start_index:placeholder_start_index + n_prompt_tokens_total_front
                ] = soft_prompt_front
                inputs_embeds[
                placeholder_end_index[idx] - n_prompt_tokens_back:placeholder_end_index[idx]] = soft_prompt_back
                input_embeds_list.append(inputs_embeds)
        inputs_embeds = torch.stack(input_embeds_list, axis=0)  # .to(model.device)
        labels = torch.stack(label_list, axis=0)  # .to(model.device)
        return inputs_embeds, labels
    def concatenate_inputs_and_labels(
            self,
            inputs_embeds_merge_acc,
            inputs_embeds_merge_rej,
            labels_acc,
            labels_rej,
    ):
        input_ids0 = torch.tensor([self.pad_token_id], dtype=torch.long).cuda()
        pad_tensor = self.get_model().embed_tokens(input_ids0).repeat(labels_acc.shape[0], 1)
        max_length = max(labels_acc.shape[1], labels_rej.shape[1])
        pad_value = -100

        if labels_acc.shape[1] == max_length:  # extend rej_labels
            pad_length = max_length - labels_rej.shape[1]
            append_tensor = torch.full((labels_rej.shape[0], pad_length), pad_value, dtype=labels_rej.dtype,
                                       device=labels_rej.device)
            labels_rej = torch.cat(
                [
                    labels_rej,
                    append_tensor,
                ],
                dim=1,
            )
            inputs_embeds_merge_rej = torch.cat(
                [inputs_embeds_merge_rej, pad_tensor.unsqueeze(1).repeat(1, pad_length, 1)], dim=1)
        elif labels_rej.shape[1] == max_length:  # extend_acc_albels
            pad_length = max_length - labels_acc.shape[1]
            append_tensor = torch.full((labels_acc.shape[0], pad_length), pad_value, dtype=labels_acc.dtype,
                                       device=labels_acc.device)
            labels_acc = torch.cat(
                [
                    labels_acc,
                    append_tensor,
                ],
                dim=1,
            )
            inputs_embeds_merge_acc = torch.cat(
                [inputs_embeds_merge_acc, pad_tensor.unsqueeze(1).repeat(1, pad_length, 1)],
                dim=1)
        labels = torch.cat((labels_acc, labels_rej), dim=0)
        input_embeds = torch.cat((inputs_embeds_merge_acc, inputs_embeds_merge_rej), dim=0)
        return input_embeds, labels
    def generate_attention_mask(self,labels):
        # 获取 labels 的形状
        N, Mi = labels.shape

        # 初始化 attention_mask，全为 1
        attention_mask = torch.ones(N, Mi, dtype=torch.float16)

        # 遍历每一行，找到 -100 第一次出现的位置，并从该位置开始填充
        for i in range(N):
            # 获取当前行
            row = labels[i]
            # 找到 -100 第一次出现的位置
            padding_start_idx = (row == -100).nonzero(as_tuple=False)
            if padding_start_idx.size(0) > 0:  # 检查是否找到了 -100
                padding_start_idx = padding_start_idx[0].item()
                # 从该位置开始，将 attention_mask 设置为 0
                attention_mask[i, padding_start_idx:] = 0

        return attention_mask
    def get_batch_logps(
            self,
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
            bn=False,
            show=False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        # print(labels.shape,logits.shape)
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        if show:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[:1].reshape(-1, 32000)
            shift_labels = labels[:1].reshape(-1)
            loss = loss_fct(shift_logits, shift_labels).detach()
            shift_logits = logits[1:2].reshape(-1, 32000)
            shift_labels = labels[1:2].reshape(-1)
            loss2 = loss_fct(shift_logits, shift_labels).detach()
            if bn:
                print('loss_fct1 for bn', loss, loss2)
            else:
                print('loss_fct1', loss, loss2)
            shift_logits = logits[4:5].reshape(-1, 32000)
            shift_labels = labels[4:5].reshape(-1)
            loss = loss_fct(shift_logits, shift_labels).detach()
            shift_logits = logits[5:6].reshape(-1, 32000)
            shift_labels = labels[5:6].reshape(-1)
            loss2 = loss_fct(shift_logits, shift_labels).detach()
            if bn:
                print('loss_fct2 for bn', loss, loss2)
            else:
                print('loss_fct2', loss, loss2)
        # exit()

        # 0.046
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0  # 只取有用的token
        # print(logits.log_softmax(-1).shape)
        # print('labels',labels)
        # print(logits.log_softmax(-1).shape,labels)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(
            2)  # 没用的就去0号位置的概率
        # print(per_token_logps.shape)
        # print(per_token_logps)
        # exit()

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)  #

    def forward_defend(self, samples,attack_embs,show=False):
        bad_samples=samples[0]
        # bn_samples=samples[1]
        batch_messages = bad_samples["malicious_input_ids"]
        batch_labels_acc = bad_samples["malicious_labels_acc"]
        batch_labels_rej = bad_samples["malicious_labels_rej"]
        inputs_embeds_merge_acc, labels_acc = self.preparing_embedding_attack(batch_messages, batch_labels_acc,attack_embs)
        inputs_embeds_merge_rej, labels_rej = self.preparing_embedding_attack(batch_messages,batch_labels_rej,attack_embs)

        inputs_embeds_merge, labels = self.concatenate_inputs_and_labels(inputs_embeds_merge_acc,
                                                                    inputs_embeds_merge_rej, labels_acc, labels_rej)
        len_chosen = inputs_embeds_merge_acc.shape[0]
        out = self.model(inputs_embeds=inputs_embeds_merge, output_hidden_states=True)
        hidden_states = out[0]
        # print('aaa',hidden_states.dtype)
        # for param in self.lm_head.parameters():
        #     print('bbb',param.dtype)
        # print('cccc',inputs_embeds_merge.dtype)
        logits = self.lm_head(hidden_states)  # [:4]

        labels = labels
        all_logps, size_completion = self.get_batch_logps(
            logits,
            labels,
            show=show
        )
        policy_chosen_logps = all_logps[:len_chosen]
        policy_rejected_logps = all_logps[len_chosen:]
        loss = self.dpo_loss(policy_rejected_logps, policy_chosen_logps, ref=True)
        return loss
    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            # reference_chosen_logps: torch.FloatTensor,
            # reference_rejected_logps: torch.FloatTensor,
            # beta: float = 0.1,
            ref=False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(policy_chosen_logps.device)
        # ref_logratios = ref_logratios.to(policy_chosen_logps.device)
        if ref is True:
            # ref_logratios = reference_chosen_logps - reference_rejected_logps
            # ref_logratios = ref_logratios.to(policy_chosen_logps.device)
            logits = pi_logratios#- ref_logratios
            losses = (
                    -F.logsigmoid(logits)#-0.1*policy_chosen_logps
            )
        elif ref is False:
            logits = pi_logratios
            losses = (
                    -F.logsigmoid(logits)#-0.1*policy_chosen_logps
            )
        elif ref=='sft':
            # print('here')
            # exit()
            losses = (
                    -policy_chosen_logps
            )
        else:
            raise ValueError
        return losses.mean()#, chosen_rewards, rejected_rewards
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_loss: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # print('hereherehereherehereherehere')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print('input2',input_ids.shape)
        # exit()
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        output_hidden_states=output_hidden_states
        outputs = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        # print('share final',logits.shape,labels.shape)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = -loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def defend_step(self,samples,attack_prompt,show):
        print('defend')
        defend_loss = self.forward_defend(samples, attack_prompt, show)  # ["loss"]
        # print(defend_loss)
        print('utility')
        utility_loss = self.forward_utility(**samples[1])  # ["loss"]
        # utility_loss = self.dpo_loss(policy_chosen_logps, policy_rej_logps, ref='sft')

        loss = utility_loss + defend_loss
        return loss
    # def forward(
    #     self,
    #     malicious_input_ids: Optional[List] = None,
    #     malicious_labels_acc: Optional[List] = None,
    #     malicious_labels_rej: Optional[List] = None,
    #     malicious_soft_id:Optional[List] = None,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     attn_loss: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     images: Optional[torch.FloatTensor] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     benign_labels=copy.deepcopy(labels)
    #
    #     batch_messages = malicious_input_ids
    #     batch_labels_acc = malicious_labels_acc
    #     batch_labels_rej = malicious_labels_rej
    #     # adv_length=16
    #     characters_set = "!@*¥(&@*¥&()ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    #     adv_length = 16
    #     attack_iter = 40
    #
    #     soft_prompt = self.get_model().embed_tokens(
    #         torch.tensor(malicious_soft_id).to(self.device)).cpu().detach()
    #     soft_prompt = nn.Parameter(soft_prompt, requires_grad=True)
    #     soft_prompt = torch.tensor(soft_prompt, dtype=torch.float32)#.to(self.device)
    #
    #
    #     optimizer_adv = torch.optim.AdamW([soft_prompt], lr=1e-3)
    #     for param in self.model.parameters():
    #         param.requires_grad = False
    #     self.model.eval()
    #     self.model.train()
    #     soft_prompt.requires_grad_(True)
    #     for i in range(attack_iter):
    #         if i == attack_iter - 2:
    #             show = True
    #         else:
    #             show = False
    #         inputs_embeds_merge_acc, labels_acc = self.preparing_embedding_attack(batch_messages, batch_labels_acc,
    #                                                                               soft_prompt)
    #
    #         inputs_embeds_merge_rej, labels_rej = self.preparing_embedding_attack(batch_messages, batch_labels_rej,
    #                                                                               soft_prompt)
    #
    #         inputs_embeds_merge, labels = self.concatenate_inputs_and_labels(inputs_embeds_merge_acc,
    #                                                                          inputs_embeds_merge_rej, labels_acc,
    #                                                                          labels_rej)
    #     # print(labels.shape)
    #         len_chosen = inputs_embeds_merge_acc.shape[0]
    #         out = self.model(inputs_embeds=inputs_embeds_merge,output_hidden_states=True)
    #         hidden_states = out[0]
    #         logits = self.lm_head(hidden_states)  # [:4]
    #         labels = labels
    #         all_logps, size_completion = self.get_batch_logps(
    #             logits,
    #             labels,
    #             show=show
    #         )
    #         policy_chosen_logps = all_logps[:len_chosen]
    #         policy_rejected_logps = all_logps[len_chosen:]
    #         loss = self.dpo_loss(policy_chosen_logps, policy_rejected_logps, ref=False)
    #         loss.backward()
    #         optimizer_adv.step()
    #         # for i in range(4):
    #         #     inputs_embeds_merge_acc[i][:16]=soft_prompt#.clone()
    #     soft_prompt.requires_grad_(False)
    #     # for name, param in self.model.named_parameters():
    #     #     print('2param', name, param.requires_grad)
    #     # exit()
    #     for name, param in self.model.named_parameters():
    #         if "lora" in name:
    #             param.requires_grad = True
    #     self.model.train()
    #     malicious_samples={"malicious_input_ids":malicious_input_ids,"malicious_labels_acc":malicious_labels_acc,"malicious_labels_rej":malicious_labels_rej}
    #     benign_samples={"input_ids":input_ids,"attention_mask":attention_mask,"labels":benign_labels,"images":images}
    #     loss = self.defend_step(samples=[malicious_samples, benign_samples],
    #                             attack_prompt=soft_prompt, show=True)
    #
    #     return CausalLMOutputWithPast(
    #             loss=loss,
    #         )

    # logits=logits,
    # past_key_values=outputs.past_key_values,
    # hidden_states=outputs.hidden_states,
    # attentions=outputs.attentions,
    def forward_utility(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_loss: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # print('hereherehereherehereherehere')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print('input2',input_ids.shape)
        # exit()
        # print('input_ids', input_ids.dtype, images.dtype, attention_mask.dtype, labels.dtype)

        # exit()
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # exit()
        output_hidden_states=output_hidden_states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # exit()
        hidden_states = outputs[0]
        # print('aaa', hidden_states.dtype)
        # for param in self.lm_head.parameters():
        #     print('bbb', param.dtype)
        # print('cccc', inputs_embeds.dtype)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        print('utility_loss',loss)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return loss

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
