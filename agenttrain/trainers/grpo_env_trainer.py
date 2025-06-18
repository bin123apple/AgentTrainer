import io
import re
import base64
import warnings
from PIL import Image
from typing import Callable, Dict, Optional, Union, Any, List, Tuple
import torch.nn.functional as F
from agenttrain.vlm_modules import VLMBaseModule
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available
)
from trl.models import create_reference_model
from transformers.utils import is_peft_available
if is_peft_available():
    from peft import PeftConfig, get_peft_model
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from agenttrain.envs.environment import Environment
from agenttrain.utils.logging_utils import print_prompt_completions_sample
from vllm import LLM, SamplingParams
import torch.distributed as dist
from agenttrain.inference.vllm_client import VLLMClient
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

# monkey patch vllm client
import trl.extras.vllm_client
trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
from agenttrain.utils.torch_ope import nanmin, nanmax
from agenttrain.prompts.system_prompts import CROP_SYSTEM_PROMPT
from agenttrain.prompts.tool_description import CROP_TOOL_DESCRIPTION
from agenttrain.prompts.tool_example import CROP_TOOL_EXAMPLE

if is_wandb_available():
    import wandb

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            reward_weights,
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            vlm_module: VLMBaseModule = None,
            freeze_vision_modules: Optional[bool] = True,
            attn_implementation: str = "flash_attention_2",
            torch_dtype: str = "bfloat16",
            **kwargs,
    ):
        self.vllm_client = None
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype

        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
            # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        
        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)
            
        # Reference model
        if is_deepspeed_zero3_enabled():
            print("Deepspeed ZeRO-3 is enabled, skipping reference model creation.")
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            print("peft_config is None, Creating reference model...")
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            print("PEFT is enabled, skipping reference model creation.")
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
            
        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                print("Setting pad_token_id and eos_token from tokenizer...")
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token = pad_token_id
                processing_class.eos_token = processing_class.tokenizer.eos_token_id
            else:
                print("Setting pad_token_id and eos_token from processing_class...")
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=None, # The adapter is already applied to the model
            **kwargs,
        )
        self.env = env
        self.scale_rewards = scale_rewards
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            repetition_penalty=self.repetition_penalty
        )
        self.reward_weights = reward_weights
        
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        # print('self._buffered_inputs:', self._buffered_inputs)
        
    def _prepare_multimodal_chat_template(self, prompts: List[str], images: List[Image.Image]) -> List[dict]:
        '''
        Prepare the multimodal chat template for vLLM inference.
        This function takes a list of prompts and a list of images, and returns a list of dictionaries
        that can be used as input to the vLLM model.
        '''
        multimodal_inputs = []
        for prompt, image in zip(prompts, images):
            # initial_prompts = CROP_SYSTEM_PROMPT.format(
            # tool_descriptions=CROP_TOOL_DESCRIPTION,
            # tool_example=CROP_TOOL_EXAMPLE
            # ) + f'\nNow please help me to identify the coordinate of the following element : \n{prompt}'
            initial_prompts = (
                "Output only the coordinate of one point in your response. "
                f"What element matches the following task: {prompt}"
            )
            if image is not None:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                initial_message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": initial_prompts},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }
                    ]
            else:
                initial_message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": initial_prompts},
                            ]
                        }
                    ]
            multimodal_inputs.append(initial_message)
        return multimodal_inputs
    
    def prepare_model_inputs(self, prompts_text, images, return_tensors="pt", 
                         padding=True, padding_side="left", add_special_tokens=False):
        if len(images) > 0:
            model_inputs = self.processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            model_inputs = self.processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        # print(f"Model inputs: {model_inputs}")
        return model_inputs
    
    def generate_logits_to_keep_batch(
        self,
        batch_input_ids,
        start_sequence,
        end_sequence
    ):
        """
        生成 logits_to_keep 掩码，标记出每条序列中位于
        start_sequence（例如 [151644, 77091, 198]）与 end_sequence（例如 [151645]）之间的所有 token。

        参数：
        - batch_input_ids: List[List[int]] 或 Tensor(N, S)，
            N 条序列，每条长度为 S。
        - start_sequence:  List[int]，表示 <|im_start|>assistant 对应的多 token 标记序列。
        - end_sequence:    List[int]，表示 <|im_end|> 对应的多 token 标记序列。

        返回：
        - List[List[int]]，大小 (N, S)，其中 1 表示该位置属于 assistant
            生成区段（不包括 start_sequence 和 end_sequence 本身），其它位置为 0。
        """
        # 如果传入的是 Tensor，就先转成 Python list
        if isinstance(batch_input_ids, torch.Tensor):
            batch_input_ids = batch_input_ids.tolist()

        mask_batch = []
        len_start = len(start_sequence)
        len_end = len(end_sequence)

        for seq in batch_input_ids:
            S = len(seq)
            mask = [0] * S
            i = 0
            in_assistant = False

            while i < S:
                # 如果当前片段匹配 start_sequence 且尚未进入 assistant 段
                if not in_assistant and i + len_start <= S and seq[i : i + len_start] == start_sequence:
                    in_assistant = True
                    i += len_start
                    continue

                # 如果当前片段匹配 end_sequence 且已在 assistant 段
                if in_assistant and i + len_end <= S and seq[i : i + len_end] == end_sequence:
                    in_assistant = False
                    i += len_end
                    continue

                # 普通 token：如果正处于 assistant 段，则打上掩码
                if in_assistant:
                    mask[i] = 1
                i += 1

            mask_batch.append(mask)
        # print(f"Generated logits_to_keep mask: {mask_batch}, {len(mask_batch[0])}")
        return mask_batch

    def compute_rewards(
        self,
        questions: List[str],
        inputs: List[Dict],
        images: List[Optional[Image.Image]],
        completion_messages: List[Dict],
        device: torch.device,
    ):
        """
        多卡环境下计算 rewards：
        - device: 当前进程所使用的设备 (e.g. torch.device("cuda", local_rank))
        - 假设已初始化 torch.distributed
        """
        mode = "eval" if self.control.should_evaluate else "train"
        completions = completion_messages
        num_samples = len(questions)
        # print(f"Computing rewards for {num_samples} samples on device {device}")
        num_funcs = len(self.reward_funcs)
        # 在当前 device 上创建张量
        rewards_per_func = torch.zeros((num_samples, num_funcs), device=device)

        # 1. 逐个 reward 函数计算
        for i, reward_func in enumerate(self.reward_funcs):
            # 抽取 inputs 中所有除 prompt/completion 以外的 key
            keys = [k for k in inputs[0].keys() if k not in ("prompt", "completion")]
            reward_kwargs = {
                key: [example[key] for example in inputs]
                for key in keys
            }
            if any(img is not None for img in images):
                reward_kwargs["images"] = [example.get("image") for example in inputs]

            # 调用 reward 函数
            out = reward_func(
                questions=questions,
                completions=completions,
                **reward_kwargs
            )
            # print(f"[{reward_func.__name__}] -> {out}")

            # None 转 NaN，并放到 device 上
            out = [r if r is not None else torch.nan for r in out]
            rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        # 2. 警告：如果某一行所有函数都返回 None
        all_nan_mask = torch.isnan(rewards_per_func).all(dim=1)
        if all_nan_mask.any():
            idx = all_nan_mask.nonzero(as_tuple=True)[0][0].item()
            bad_kwargs = {k: v[idx] for k, v in reward_kwargs.items()}
            bad_kwargs["prompt"] = questions[idx]
            bad_kwargs["completion"] = completions[idx]
            if any(img is not None for img in images):
                bad_kwargs["image"] = images[idx]
            warnings.warn(
                f"All reward functions returned None for sample {idx}, kwargs: {bad_kwargs!r}"
            )

        # 3. 多卡 all_gather
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(rewards_per_func) for _ in range(world_size)]
            dist.all_gather(gathered, rewards_per_func)
            # 拼接成 (world_batch_size, num_funcs)
            rewards_per_func = torch.cat(gathered, dim=0)
        
        # log to wandb  
        for i, reward_func_name in enumerate(self.reward_func_names):
            if reward_func_name in ('correct_crop_func', 'correct_answer_reward_func'):
                mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)

        # 4. 应用权重并求和
        weights = torch.tensor(self.reward_weights, device=device).unsqueeze(0)  # (1, num_funcs)
        final_rewards = (rewards_per_func * weights).nansum(dim=1)          # (world_batch_size,)

        # print(f"[Weights] {weights}")
        # print(f"[Final Rewards] {final_rewards}")
        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)

        return final_rewards

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]):
        '''
        Prepare several things:
        1. prompt_completion_ids: (B, L)
        2. logits_to_keep: (B, L)
        3. reward
        For further loss calculation
        args:
            input: dataset dict, each element is a dict with keys:
                - "question": str, the prompt question
                - "image": PIL.Image or None, the image if available
            output: A dict with keys:
                - "prompt_completion_ids": torch.Tensor, shape (B, L)
                - "logits_to_keep": torch.Tensor, shape (B, L)
                - "rewards"
        '''
        mode = "eval" if self.control.should_evaluate else "train"
        # prepare hardware device
        device = self.accelerator.device
        
        # prepare inference data
        prompts = [x["question"] for x in inputs]
        # print(f"Prompts size: {len(prompts)}")
        answers = [x["answer"] for x in inputs] if "answer" in inputs[0] else None
        images = [Image.open(io.BytesIO(x.get("image"))) for x in inputs]
        
        # upload the new model weights to vllm
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        
        # prepare chat template for vllm inference
        multimodal_inputs = self._prepare_multimodal_chat_template(prompts, images)
        
        # gather all prompts and images from all processes, env step
        all_multimodal_inputs = gather_object(multimodal_inputs)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_multimodal_inputs, 
                llm=self.vllm_client, 
                sampling_params=self.sampling_params,
            )
            all_prompts = env_result['all_prompts'] # calculate log_pb
            all_images = env_result['images'] # calculate log_pb
            all_messages = env_result['all_messages'] # calculate rewards
        else:
            # Non main processes will wait for the main process to finish
            all_prompts = [None] * len(all_multimodal_inputs)
            all_images = [None] * len(all_multimodal_inputs)
            all_messages = [None] * len(all_multimodal_inputs)
            
        all_prompts = broadcast_object_list(all_prompts, from_process=0)
        all_images = broadcast_object_list(all_images, from_process=0)
        all_messages = broadcast_object_list(all_messages, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        ) # len(prompts) 是每个进程的本地 batch 大小
        # Back to each process's local batch size
        all_prompts = all_prompts[process_slice]
        all_images = all_images[process_slice]
        all_messages = all_messages[process_slice]
        
        # OK, now let's use processor to get the input_ids(Will be paded in processor)/attention_mask/pixel_values/image_grid_thw
        # convert prompts to correct format
        all_prompts_one_image_pad = []
        for prompt in all_prompts:
            prompt = re.sub(r'(?:<\|image_pad\|>)+', '<|image_pad|>', prompt)
            all_prompts_one_image_pad.append(prompt)
            
        # debug
        total_all_prompts_pads = sum(prompt.count('<|image_pad|>') for prompt in all_prompts)
        # print(f"Total number of <|image_pad|> tokens in all prompts: {total_all_prompts_pads}")
        total_all_prompts_pads_one_image = sum(prompt.count('<|image_pad|>') for prompt in all_prompts_one_image_pad)
        # print(f"Total number of <|image_pad|> tokens in all prompts (one image pad): {total_all_prompts_pads_one_image}")
        
        
        model_inputs = self.prepare_model_inputs(prompts_text = all_prompts_one_image_pad, 
                                                images = all_images, return_tensors="pt", padding=True, 
                                                padding_side="left", add_special_tokens=False)
        model_inputs = Trainer._prepare_inputs(self, model_inputs)

        input_ids = model_inputs["input_ids"]
        
        #debug
        mask = input_ids == 151655
        count = int(mask.sum().item())
        # print(f"Token ID 151655 出现了 {count} 次")
        
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        image_grid_thw = model_inputs["image_grid_thw"]
        # print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}"
            #  f", Pixel values shape: {pixel_values.shape}, Image grid shape: {image_grid_thw.shape}")

        logits_to_keep = self.generate_logits_to_keep_batch(
            input_ids, 
            start_sequence=[151644, 77091, 198], #<|im_start|>assistant\n
            end_sequence=[198, 151644, 872, 198] #\n<|im_start|>user\n
        )
        
        # Compute the old_per_token_logps & ref_per_token_logps
        with torch.no_grad():
            # print(f'num_iterations: {self.num_iterations}')
            if self.num_iterations > 1:
                old_per_token_logps,_ = self._get_per_token_logps(
                    self.model, 
                    input_ids = input_ids,         # Tensor of shape (B, S)
                    attention_mask = attention_mask,    # Tensor of shape (B, S)
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    logits_to_keep = logits_to_keep
                )
                # print(f"old_per_token_logps: {old_per_token_logps}, shape: {old_per_token_logps.shape}")
            else:
                # print("No old_per_token_logps to compute, using None.")
                old_per_token_logps = None

            # （2）如果 beta>0 且给了一个 ref_model，需要对应计算参考模型 log‐probs
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps,_ =self._get_per_token_logps(
                    self.ref_model, 
                    input_ids = input_ids,         # Tensor of shape (B, S)
                    attention_mask = attention_mask,    # Tensor of shape (B, S)
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    logits_to_keep = logits_to_keep
                )
            else:
                # 关闭 adapter 时当基线模型
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps,_ = self._get_per_token_logps(
                        self.model, 
                        input_ids = input_ids,         # Tensor of shape (B, S)
                        attention_mask = attention_mask,    # Tensor of shape (B, S)
                        pixel_values = pixel_values,
                        image_grid_thw = image_grid_thw,
                        logits_to_keep = logits_to_keep
                    )
        
        # reward fuction test and logic
        inputs = [{"task": "vg", "answer": answer} for answer in answers]
        rewards = self.compute_rewards(questions = prompts, inputs = inputs, 
                                       images = all_images, 
                                       completion_messages = all_messages, 
                                       device = device)
        
        # print(f"rewards before gather: {rewards}, shape: {rewards.shape}")
        # rewards = gather(rewards)
        # print(f"rewards after gather: {rewards}, shape: {rewards.shape}")
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # 计算组内reward均值 (B/n, n,)
        # print(f"mean_grouped_rewards: {mean_grouped_rewards}, shape: {mean_grouped_rewards.shape}")
        # mean_grouped_rewards.shape == (world_batch_size / num_generations,)

        # 把每组平均值 repeat_interleave 回到 (world_batch_size,) 
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # print(f"mean_grouped_rewards: {mean_grouped_rewards}")
        advantages = (rewards - mean_grouped_rewards)
        # print(f"advantages: {advantages}")
        
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # 计算组内reward标准差
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        # self._metrics[mode]["advantages"].append(advantages)
        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        # print(f"advantages after slice: {advantages}, shape: {advantages.shape}")
        
        # Log the metrics
        self._metrics[mode]["reward/mean"].append(mean_grouped_rewards.mean().item())
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask, 
            "pixel_values": pixel_values, 
            "image_grid_thw": image_grid_thw,
            "logits_to_keep": logits_to_keep, 
            "advantages": advantages, 
            "ref_per_token_logps": ref_per_token_logps,
            "old_per_token_logps": old_per_token_logps,
        }

    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def selective_log_softmax(self, logits, index):
        """
        A memory-efficient implementation of the common `log_softmax -> gather` operation.

        This function is equivalent to the following naive implementation:
        ```python
        logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        ```

        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`.
            index (`torch.Tensor`):
                Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

        Returns:
            `torch.Tensor`:
                Gathered log probabilities with the same shape as `index`.
        """
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def _get_per_token_logps(
        self,
        model,
        input_ids,         # Tensor of shape (B, S)
        attention_mask,    # Tensor of shape (B, S)
        pixel_values,
        image_grid_thw,    
        logits_to_keep,    # List[List[int]] of length B, each inner list length S (0/1 mask)
        batch_size=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从完整 input_ids 里，提取 logits_to_keep 标记位置对应的 token
        的 log‐probs 并返回一个 (B, L_max) 的 tensor，其中 L_max 是本批
        中单条样本被标记位置总数的最大值。输出中每行对应一条样本，
        在被标记（值为1）的那些位置上存 log‐prob，未被标记的位置用0填充。
        
        参数：
        - model:           返回 logits 的 HuggingFace 模型
        - input_ids:       Tensor(B, S)，包含用户+助手所有 token 序列
        - attention_mask:  Tensor(B, S)，同样对应 input_ids
        - logits_to_keep:  List[List[int]]，大小 (B, S)，元素 0/1。1 表示该位置
                            属于助手生成内容，需要计算 log‐prob；0 表示跳过。
        - batch_size:      int，可选。若不为 None，则按子批大小分块计算，减少显存峰值。
        
        返回：
        - Tensor of shape (B, L_max)，每行代表一条样本中“标记 = 1”的那些 token
            的 log‐probs，左侧用 0 填充到相同宽度 L_max。
        """
        B, S = input_ids.size()
        # print(">>> get_per_token_logps input_ids shape:", input_ids.shape)
        batch_size = batch_size or B
        all_logps = []  # 用来存储每个 batch 的 log‐probs
        logits_to_keep = torch.tensor(logits_to_keep, dtype=torch.long, device=input_ids.device)
        # print(">>> logits_to_keep shape:", logits_to_keep, logits_to_keep.shape)
        
        for i in range(0, B, batch_size):
            input_ids_batch      = input_ids[i : i + batch_size]       # (B_chunk, S)
            attention_batch      = attention_mask[i : i + batch_size]  # (B_chunk, S)
            mask_batch           = logits_to_keep[i : i + batch_size]  # List length B_chunk, each is list of length S
            # pixel_values_batch=pixel_values[i : i + batch_size]
            # image_grid_thw_batch=image_grid_thw[i : i + batch_size]

            B_chunk, _ = input_ids_batch.shape
            # 1) 前向计算整条 seq 的 logits （(B_chunk, S, V)）
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_batch,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            # print(">>> outputs:", outputs)
            full_logits = outputs.logits  # (B_chunk, S, V)
            # print(">>> full_logits:", full_logits, full_logits.shape)
            full_logits = full_logits[:, :-1, :] # Change shape to (B_chunk, S-1, V), del the last token logits
            input_ids_batch = input_ids_batch[:, 1: ]  # (B_chunk, S-1), del the first token input_ids
            logits_to_keep = logits_to_keep[:, 1: ]  # (B_chunk, S-1), do the same thing as input_ids_batch
            keep_mask          = logits_to_keep.bool()    # → (B, S-1)

            # 2) 计算哪些列（时间步）至少有一个样本需要保留
            col_mask = keep_mask.any(dim=0)               # → (S-1,), dtype=torch.bool

            # 3) 用这个列掩码同步裁剪三者
            logits_to_keep = logits_to_keep[:, col_mask]  # → (B, S',)
            logits       = full_logits   [:, col_mask, :]  # → (B, S', V)
            # print(">>> logits after col_mask:", logits, logits.shape)
            input_ids_batch    = input_ids_batch[:, col_mask]     # → (B, S')
            # print(">>> input_ids_batch after col_mask:", input_ids_batch, input_ids_batch.shape)
            keep_mask    = keep_mask     [:, col_mask]     # → (B, S')
            # print(">>> keep_mask after col_mask:", keep_mask, keep_mask.shape)
            

            # 4) 计算批次级别的 log‐probs：传入 (B_chunk, L_max, V) 和 (B_chunk, L_max)
            #    selective_log_softmax 会对每个 batch 中的行分别计算
            single_logps_batch = self.selective_log_softmax(
                logits,    # (B_chunk, L_max, V)
                input_ids_batch        # (B_chunk, L_max)
            )  # 返回 (B_chunk, L_max)
            all_logps.append(single_logps_batch)

        result = torch.cat(all_logps, dim=0)  # (B, L_max_overall)
        # print(">>> all log-prob:", result)
        return result, logits_to_keep

    def compute_loss(self, model, inputs, num_items_in_batch=None):  
        # print(">> buffered before:", self._buffered_inputs)
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
        
        advantages = inputs["advantages"]  # (B, )
        ref_per_token_logps = inputs["ref_per_token_logps"]  # (B, L_max)
        old_per_token_logps = inputs["old_per_token_logps"]  # (B, L_max)
        
        # Compute the per-token log probabilities for the model
        per_token_logps,completion_mask = self._get_per_token_logps(
            model, 
            input_ids = inputs.get("input_ids"),         # Tensor of shape (B, S)
            attention_mask = inputs.get("attention_mask"),    # Tensor of shape (B, S)
            pixel_values = inputs.get("pixel_values"),
            image_grid_thw = inputs.get("image_grid_thw"),
            logits_to_keep = inputs.get("logits_to_keep")
        )
        # print(f"current per_token_logps: {per_token_logps}, shape: {per_token_logps.shape}")
        # print(f"completion_mask: {completion_mask}, shape: {completion_mask.shape}")

        # Compute the KL divergence between the model and the reference model
        # https://arxiv.org/abs/2402.03300 Eq (4)
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # print(f"per_token_kl: {per_token_kl}, shape: {per_token_kl.shape}")
        
        # Compute the loss
        # https://arxiv.org/abs/2402.03300 Eq (3)
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # print(f"coef_1: {coef_1}, shape: {coef_1.shape}")
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # print(f"coef_2: {coef_2}, shape: {coef_2.shape}")
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # print(f"per_token_loss1: {per_token_loss1}, shape: {per_token_loss1.shape}")
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        # print(f"per_token_loss2: {per_token_loss2}, shape: {per_token_loss2.shape}")
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # print(f"per_token_loss: {per_token_loss}, shape: {per_token_loss.shape}")
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            # print("completion_mask shape and sum:", completion_mask.shape, completion_mask.sum(-1))
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


