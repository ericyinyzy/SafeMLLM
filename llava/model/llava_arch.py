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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)

            # self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            # print(self.mm_projector.weight)
            # exit()
            self.mm_projector = build_vision_projector(config)
            # print(self.mm_projector.weight)
            # exit()
        # print('1111111',self.vision_tower)
        # exit()


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        # print('hereherehere',vision_tower)
        # exit()
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]

        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # print('here_initilize')
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)


        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower
        # print('222222222',self.vision_tower)

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        # print('hahahah')
        # if not hasattr(self, 'mm_projector'):
        #     print('is it here?')
        #     self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        # exit()
        # if pretrain_mm_mlp_adapter is not None:
        #     # print(pretrain_mm_mlp_adapter)
        #     # exit()
        #     mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        #     # print(mm_projector_weights)
        #     # exit()
        #     # exit()
        #
        #     def get_w(weights, keyword):
        #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        #
        #     self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        # exit()


        # if pretrain_mm_mlp_adapter is not None:
        #     # print('bbbbbbbbb')
        #     # print(pretrain_mm_mlp_adapter)
        #     mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        #     # print('here',mm_projector_weights)
        #     # for k, v in mm_projector_weights.items():
        #     #     print(k,v.shape)
        #     # exit()
        #     # print('there',self.mm_projector)
        #     # exit()
        #     def get_w(weights, keyword):
        #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        #
        #     self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        # print('llava1',self.model)
        # exit()


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        # print(images.dtype)
        image_features = self.get_model().get_vision_tower()(images)
        # print(image_features.device,self.get_model().mm_projector.weight.device)

        # exit()
        # print('qnmgb',image_features.dtype)
        # for param in self.get_model().mm_projector.parameters():
        #     print('param',param.dtype)
        # exit()

        image_features = self.get_model().mm_projector(image_features)

        return image_features
    def find_kwd_id(self,ans_id,slogan_th_id_list):
        # car
        # subsequences = [torch.tensor([350, 25365]).cuda(), torch.tensor([1559]).cuda(), torch.tensor([19716]).cuda(),
        #                torch.tensor([20134, 29963]).cuda(), torch.tensor([26544]).cuda(),
        #                torch.tensor([534, 2707]).cuda(), torch.tensor([1444, 1022]).cuda(), torch.tensor([1109]).cuda(),
        #                torch.tensor([18647]).cuda(), torch.tensor([24413]).cuda(), torch.tensor([7048, 550]).cuda(),
        #                torch.tensor([8818, 29875]).cuda()]
        # laptop
        # subsequences = [torch.tensor([19022]).cuda(), torch.tensor([2011, 519, 6601]).cuda(),
        #                 torch.tensor([1461, 29882, 2495, 6601]).cuda(),
        #                 torch.tensor([23012, 29899, 3332, 6601]).cuda(), torch.tensor([19022, 6601]).cuda(),
        #                 torch.tensor([425, 415, 3554]).cuda(), torch.tensor([19022, 23226]).cuda(),
        #                 torch.tensor([4326, 10967]).cuda()]
        #sandwich
        subsequences = [torch.tensor([11982, 16416]).cuda(), torch.tensor([6866, 914]).cuda(),
                        torch.tensor([298, 1117, 26120]).cuda(),
                        torch.tensor([12244]).cuda(),
                        torch.tensor([7243, 2172]).cuda(), torch.tensor([26072, 414]).cuda(),
                        torch.tensor([298, 1117, 2007, 414]).cuda(),
                        torch.tensor([7243, 262, 275]).cuda(), torch.tensor([11463, 567]).cuda(),
                        torch.tensor([15612, 3780, 2272]).cuda()]

        # kwd_id_list=[]
        st_list=torch.where(ans_id == 2)[0].cpu().tolist()
        # print(st_list)
        # exit()
        st_list=[0]+st_list
        # print(st_list)

        # print('slogan_th_id_list',slogan_th_id_list,st_list)
        # print('ans_id',ans_id)

        kwd_id_list=[]
        start_search_idx=None

        for idx,slogan_th_id in enumerate(slogan_th_id_list):
            slogan_st_id=slogan_th_id
            positions = []
            tensor_length = ans_id.size(0)
            # if idx>0:
            for (a, b) in zip(st_list[:-1], st_list[1:]):
                if slogan_th_id > a and slogan_st_id < b:
                    start_search_idx = a
            if start_search_idx is None:
                print('wrong here,')
                print(slogan_th_id,st_list,ans_id)
                start_search_idx=0
            for subseq in subsequences:
                sub_length = subseq.size(0)
                # found = False
                for start_idx in range(start_search_idx,tensor_length - sub_length + 1):
                    if start_idx>=slogan_st_id:
                        break
                    if torch.equal(ans_id[start_idx:start_idx + sub_length], subseq):
                        positions.extend([i for i in range(start_idx, start_idx + sub_length)])
                    # found = True
                    # break
            kwd_id_list.append(positions)
            # start_search_idx=
            # print('kwd_id_list',kwd_id_list)


            # if not found:
            #     positions.append(None)  # Or some placeholder to indicate not found
        # exit()
        return kwd_id_list
    def find_slogan_id(self,ans_id,subsequence=torch.tensor([8653,6781,  3322, 964]).cuda()):
        # laptop = torch.tensor([853, 280, 1161, 6760, 28157]
        # car: torch.tensor([22850,  3575,  8373]).cuda()
        #sandwich torch.tensor([6781,  3322, 964]).cuda()
        sub_len = len(subsequence)
        slogan_id_list=[]
        for i in range(len(ans_id) - sub_len + 1):
            if torch.equal(ans_id[i:i + sub_len], torch.tensor(subsequence)):
                slogan_id_list.append(i-1)
                # return [i, i + sub_len - 1]
        return slogan_id_list

    def prepare_inputs_labels_for_multimodal_attack(
            self, input_ids, attention_mask, past_key_values, labels, images,model,lm_head
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
        #     loss=-torch.sum(image_features)
        #     loss.backward()
        #     print(images.grad)
        # exit()

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # print(image_features.shape)
        all_slogan_tks = []
        all_kwd_tks = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            slogan_tks = []
            # print(cur_input_ids.device)
            # exit()
            kwd_tks = []
            # print('cur_',cur_input_ids.shape)
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # print(batch_idx,'aaaaaa')
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                            0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # print('image_token_indices1', image_token_indices)
            # exit()
            # print()
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while image_token_indices.numel() > 0:
                # print('cur_image_idx',cur_image_idx)
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    # print('hereeeeeee')
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                    # print('image_token_indices1', image_token_indices)
                else:
                    # print('thereeeeeee')
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    # print(cur_input_ids[:image_token_start].device,self.get_model().embed_tokens(cur_input_ids[:image_token_start]).device)
                    # print(cur_input_ids.device)
                    # exit()
                    # print(image_token_indices.numel(),self.get_model().embed_tokens(cur_input_ids[:image_token_start]).shape,cur_image_features.shape)
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                # exit()
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                # print('image_token_indices2',image_token_indices)

            # print('cur_input_ids',cur_input_ids.shape,cur_input_ids)
            previous_length = 0
            for id, ele in enumerate(cur_new_input_embeds):
                previous_length += ele.shape[0]
                # print(id, ele.shape)
            # print(cur_input_ids.shape)
            ################
            ####################

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    # print('heeeeeeee')
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    # print('sheeeeeeeee')
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            # exit()
            # print(len(cur_new_input_embeds))
            # for i in cur_new_input_embeds:
            #     print(i.shape)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            # for i in cur_new_input_embeds:
            #     print('currrrrrrent',i,i.shape)
            # print('cur_new_input_embeds',batch_idx,cur_new_input_embeds.shape)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

            # exit()
        # print(new_input_embeds[0].shape,len(new_input_embeds))
        # for x in new_input_embeds:
        #     print(x.shape)
        # exit()
        # print('slogan_tks', all_slogan_tks,all_kwd_tks)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # print('cccccddddd')
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            # print('dddddddcccc')
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            # print(new_input_embeds.shape)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        # print(new_input_embeds.shape)
        labels=new_labels
        # output_hidden_states = output_hidden_states
        # inputs_embeds = inputs_embeds.clone().detach().to(torch.float).requires_grad_(True)
        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=new_input_embeds,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # print('output',len(outputs),output_attentions)
        # exit()
        hidden_states = outputs[0]

        logits = lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        return shift_logits,shift_labels
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images
    ):
        # print('input',input_ids.shape)
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            # print('return here',vision_tower,images,input_ids.shape)
            # exit()
            return input_ids, attention_mask, past_key_values, None, labels
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
        #     loss=-torch.sum(image_features)
        #     loss.backward()
        #     print(images.grad)
        # exit()

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # print(image_features.shape)
        all_slogan_tks = []
        all_kwd_tks = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            slogan_tks = []
            # print(cur_input_ids.device)
            # exit()
            kwd_tks = []
            # print('cur_',cur_input_ids.shape)
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # print(batch_idx,'aaaaaa')
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                            0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # print('image_token_indices1', image_token_indices)
            # exit()
            # print()
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                # print('cur_image_idx',cur_image_idx)
                # print(image_features.shape)
                # exit()
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    # print('hereeeeeee')
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                else:
                    # print('thereeeeeee')
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    # print(cur_input_ids[:image_token_start].device,self.get_model().embed_tokens(cur_input_ids[:image_token_start]).device)
                    # print(cur_input_ids.device)
                    # exit()
                    # print(image_token_indices.numel(),self.get_model().embed_tokens(cur_input_ids[:image_token_start]).shape,cur_image_features.shape)
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                # exit()
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                # print('image_token_indices2',image_token_indices)

            # print('cur_input_ids',cur_input_ids.shape,cur_input_ids)
            previous_length = 0
            for id, ele in enumerate(cur_new_input_embeds):
                previous_length += ele.shape[0]
                # print(id, ele.shape)
            # print(cur_input_ids.shape)
            ################
            ####################

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    # print('heeeeeeee')
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    # print('sheeeeeeeee')
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            # exit()
            # print(len(cur_new_input_embeds))
            # for i in cur_new_input_embeds:
            #     print(i.shape)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            # for i in cur_new_input_embeds:
            #     print('currrrrrrent',i,i.shape)
            # print('cur_new_input_embeds',batch_idx,cur_new_input_embeds.shape)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

            # exit()
        # print(new_input_embeds[0].shape,len(new_input_embeds))
        # for x in new_input_embeds:
        #     print(x.shape)
        # exit()
        # print('slogan_tks', all_slogan_tks,all_kwd_tks)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # print('cccccddddd')
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            # print('dddddddcccc')
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            # print(new_input_embeds.shape)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        # print(new_input_embeds.shape)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
