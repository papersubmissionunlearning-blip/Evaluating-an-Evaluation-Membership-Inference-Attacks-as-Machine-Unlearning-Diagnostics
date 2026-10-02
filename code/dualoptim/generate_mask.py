"""Saliency-mask generation for SalUn.

`save_gradient_ratio` below is copied verbatim from upstream
`ImageClassification/generate_mask.py` (lines 14-103). Upstream's `main()` is
not vendored: it only wires up their CLI and their dataset stack, both of
which the host project replaces.

Upstream writes one mask per threshold in {0.1 ... 1.0} as `with_{t}.pt`.
The paper's CIFAR-10 SalUn configuration uses `with_0.5.pt`.
"""
import os

import torch


def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}
    retain_gradients = {}

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0.0
        retain_gradients[name] = 0.0

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data / len(forget_loader)

    for i, (image, target) in enumerate(retain_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    retain_gradients[name] += param.grad.data / len(retain_loader)

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs(gradients[name]) - torch.abs(
                retain_gradients[name]
            )

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(args.save_dir, "with_{}.pt".format(i)))
