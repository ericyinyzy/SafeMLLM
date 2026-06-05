import torch
import torch.nn as nn

from ..attack import Attack
from llava_llama_2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_llama_2.mm_utils import tokenizer_image_token


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, nprompt=1, random_start=True,attack_prompt=None,tokenizer=None):
        super().__init__("PGD", model)
        self.model=model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.nprompt = nprompt
        self.attack_prompt=attack_prompt
        self.tokenizer=tokenizer


    def forward(self, images,labels):
        r"""
        Overridden.
        """
        # print('start')
        #images = images.clone().detach().to(self.device)
        images_ = []
        adv_images_ = []
        # print(len(images))
        # exit()

        for image in images:
            image = image.clone().detach().to(self.device)
            adv_image = image.clone().detach().to(self.device)
            images_.append(image)
            adv_images_.append(adv_image)
        # print(len(images_),self.targeted,self.random_start)
        # exit()

        # if self.targeted:
        #     target_labels = labels

        #loss = nn.CrossEntropyLoss()
        loss = nn.CrossEntropyLoss(ignore_index=-200)

        #adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        universal = 1 #universal noise
        if universal == 1:
            noise = torch.zeros(1, 3, 336, 336).to(self.device)
            #print(noise)

        # exit()
        for st in range(self.steps):
            cost_step = 0
            for k in range(len(images_)):
                image = images_[k]

                image_ = image.clone()
                # print('nprompt',self.nprompt)
                for p in range(self.nprompt):
                    # print('p',p)
                    # print('current',p)

                    image_.requires_grad = True
                    # inp = []
                    if universal == 1:
                        adv_image = image_ + noise  #universal noise
                    # print(adv_image)

                    # inp.append(adv_image)
                    # inp.append(p)
                    # print(adv_image.shape)
                    input_prompt=[self.attack_prompt[p][0]]
                    target=[self.attack_prompt[p][1]]
                    images=adv_image
                    samples=self.attack_loss(input_prompt,images,target)
                    # print(samples['images'].shape,samples['labels'].shape,samples['input_ids'].shape,samples['attention_mask'].shape)
                    # exit()

                    # print(samples['images'].dtype)
                    # exit()

                    # exit()
                    # exit()
                    #
                    # samples = {
                    #     'image': adv_image,
                    #     'text_input': [self.attack_prompt[p][0]],
                    #     'text_output': [self.attack_prompt[p][1]]
                    # }

                    cost = self.get_logits(samples).loss
                    # print(cost.keys())
                    # exit()

                    # print(cost, adv_image.shape)
                    # Update adversarial images
                    # print('forwardcomplete')
                    grad = torch.autograd.grad(
                        cost, adv_image, retain_graph=False, create_graph=False#,allow_unused=True
                    )[0]

                    # print('grad',grad)

                    cost_step += cost.clone().detach()


                    adv_image = adv_image.detach() + self.alpha * grad.sign()
                    delta = torch.clamp(adv_image - image, min=-self.eps, max=self.eps)
                    adv_image = torch.clamp(image + delta, min=0, max=1).detach()

                    if universal == 1:
                        noise = adv_image - image #universal noise

            print('step: {}: {}'.format(st, cost_step))


        if universal == 1: #universal noise
            images_outputs_ = []
            for k in range(len(images_)):
                delta = torch.clamp(noise, min=-self.eps, max=self.eps)
                adv_image = torch.clamp(images_[k] + delta, min=0, max=1).detach()
                images_outputs_.append(adv_image.detach())

            return images_outputs_
    def attack_loss(self, prompts, images, targets):
        # print(prompts,targets)
        # exit()
        context_length=[]
        context_input_ids=[]
        # print(prompts,targets)
        for prompt in prompts:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()

            context_input_ids.append(input_ids)
            context_length.append(input_ids.shape[1])

        batch_size = len(targets)

        if len(context_input_ids) == 1:
            context_length = context_length * batch_size
            context_input_ids = context_input_ids * batch_size

        images = images.repeat(batch_size, 1, 1, 1)

        assert len(context_input_ids) == len(targets), f"Unmathced batch size of prompts and targets {len(context_input_ids)} != {len(targets)}"


        to_regress_tokens = [ torch.as_tensor([item[1:]]).cuda() for item in self.tokenizer(targets).input_ids] # get rid of the default <bos> in targets tokenization.
        seq_tokens_length = []
        labels = []
        input_ids = []
        # print('here',context_input_ids[0].cpu().numpy().tolist())
        # exit()
        place_holder_start=context_input_ids[0].cpu().numpy().tolist()[0].index(-200)+3

        for i, item in enumerate(to_regress_tokens):

            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)
            context_mask = torch.full([1, place_holder_start], -100,
                                      dtype=to_regress_tokens[0].dtype,
                                      device=to_regress_tokens[0].device)
            # context_mask = torch.full([1, context_input_ids[i].shape[1]], -100,
            #                           dtype=to_regress_tokens[0].dtype,
            #                           device=to_regress_tokens[0].device)
            # print(place_holder_start, context_input_ids[i], context_mask)
            labels.append( torch.cat( [context_mask,context_input_ids[i][:,place_holder_start:], item], dim=1 ) )
            # labels.append(torch.cat([context_mask, item], dim=1))
            input_ids.append( torch.cat( [context_input_ids[i], item], dim=1 ) )
            # print(labels,input_ids,item)
        # exit()

        # padding token
        pad = torch.full([1, 1], 0,
                         dtype=to_regress_tokens[0].dtype,
                         device=to_regress_tokens[0].device).cuda() # it does not matter ... Anyway will be masked out from attention...


        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):
            num_to_pad = max_length - seq_tokens_length[i]

            padding_mask = (
                torch.full([1, num_to_pad], -100,
                       dtype=torch.long,
                       device=self.device)
            )
            labels[i] = torch.cat( [labels[i], padding_mask], dim=1 )

            input_ids[i] = torch.cat( [input_ids[i],
                                       pad.repeat(1, num_to_pad)], dim=1 )
            attention_mask.append( torch.LongTensor( [ [1]* (seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        labels = torch.cat( labels, dim=0 ).cuda()
        input_ids = torch.cat( input_ids, dim=0 ).cuda()
        # print(labels)
        # print(input_ids)
        # exit()
        attention_mask = torch.cat(attention_mask, dim=0).cuda()
        return {'input_ids':input_ids,'attention_mask':attention_mask,'return_dict':True,'labels':labels,'images':images}


