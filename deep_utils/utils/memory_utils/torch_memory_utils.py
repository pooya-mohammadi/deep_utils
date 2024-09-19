from typing import Tuple, Union
import subprocess as sp
import torch


class MemoryUtilsTorch:
    @staticmethod
    def estimate_memory(
            model, sample_input, optimizer_type=torch.optim.Adam, use_amp=False, device: Union[str, int] = 0
    ):
        """Predict the maximum memory usage of the model.
        Args:
            optimizer_type (Type): the class name of the optimizer to instantiate
            model (nn.Module): the neural network model
            sample_input (torch.Tensor): A sample input to the network. It should be
                a single item, not a batch, and it will be replicated batch_size times.
            use_amp (bool): whether to estimate based on using mixed precision
            device (torch.device): the device to use
        """
        # Reset model and optimizer
        model = model.cpu()
        optimizer = optimizer_type(model.parameters(), lr=0.001)
        a = torch.cuda.memory_allocated(device)
        model.to(device)
        b = torch.cuda.memory_allocated(device)
        model_memory = b - a
        # model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
        model(sample_input.to(device))
        c = torch.cuda.memory_allocated(device)
        if use_amp:
            amp_multiplier = 0.5
        else:
            amp_multiplier = 1
        forward_pass_memory = (c - b) * amp_multiplier
        gradient_memory = model_memory
        if isinstance(optimizer, torch.optim.Adam):
            o = 2
        elif isinstance(optimizer, torch.optim.RMSprop):
            o = 1
        elif isinstance(optimizer, torch.optim.SGD):
            o = 0
        else:
            raise ValueError(
                "Unsupported optimizer. Look up how many moments are"
                + "stored by your optimizer and add a case to the optimizer checker."
            )
        gradient_moment_memory = o * gradient_memory
        total_memory = (
                model_memory
                + forward_pass_memory
                + gradient_memory
                + gradient_moment_memory
        )

        return total_memory

    @staticmethod
    def memory_test(
            model,
            optimizer_type=torch.optim.Adam,
            input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
            use_amp=False,
            device=0,
            get_extra_info=True,
    ):
        assert device != "cpu", "Only works for cpu"
        sample_input = torch.randn(
            input_size,
            dtype=torch.float32,
        ).to(device)
        max_mem_est = MemoryUtilsTorch.estimate_memory(
            model, sample_input, optimizer_type=optimizer_type, use_amp=use_amp, device=device
        )
        print("Maximum Memory Estimate MB", round(max_mem_est / (1024 ** 2), 2))
        if get_extra_info:
            optimizer = optimizer_type(model.parameters(), lr=0.001)
            print(
                "Beginning mem:",
                torch.cuda.memory_allocated(device),
                "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number",
            )
            model.to(device)
            print("After model to device:", torch.cuda.memory_allocated(device))
            for i in range(3):
                print("Iteration", i)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    a = torch.cuda.memory_allocated(device)
                    out = model(
                        sample_input.to(device)
                    ).sum()  # Taking the sum here just to get a scalar output
                    b = torch.cuda.memory_allocated(device)
                print("1 - After forward pass",
                      torch.cuda.memory_allocated(device))
                print("2 - Memory consumed by forward pass", b - a)
                out.backward()
                print("3 - After backward pass",
                      torch.cuda.memory_allocated(device))
                optimizer.step()
                print("4 - After optimizer step",
                      torch.cuda.memory_allocated(device))

    @staticmethod
    def get_gpu_memory(device: int = None, consumed: bool = False):
        if consumed:
            command = "nvidia-smi --query-gpu=memory.used --format=csv"
        else:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        if device:
            return memory_free_values[device]
        return memory_free_values
