import io
import base64
import warnings
from PIL import Image
from typing import Callable, Optional, Union, Any, List
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
from envs.environment import Environment
from utils.logging_utils import print_prompt_completions_sample
from vllm import LLM, SamplingParams
from inference.vllm_client import VLLMClient
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

# monkey patch vllm client
import trl.extras.vllm_client
trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
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

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
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
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
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
                pad_token_id = processing_class.pad_token_id

        # self.vlm_module.post_model_init(model, processing_class)
        # self.vlm_module.post_model_init(self.ref_model, processing_class)
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
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
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        ############################################
        # 1. 准备硬件设备、提取 prompt 与 image  #
        ############################################
        device = self.accelerator.device
        # print(f'Keys in in put:{inputs[0].keys()}')
        prompts = [x["question"] for x in inputs]      # inputs[i]["question"] 是一个聊天历史的 message dict 列表
        images = [Image.open(io.BytesIO(x.get("image"))) for x in inputs]    # inputs[i] 中如果有 "image"，就拿出来，否则 None
        
        ##################################################
        # 2. 用 maybe_apply_chat_template 构造文本提示， #
        #    支持多模态也照常把文本部分拼好。         #
        ##################################################
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["question"]
            for example in inputs
        ]
        #   - 这里的 maybe_apply_chat_template 会把 `example["prompt"]` 
        #     （一个 List[Dict{role,content}]）渲染成 LLM 可读的字符串，
        #     同时保留角色信息、特殊标记等。和纯文本版完全一致。

        ##################################################################
        # 3. 调用 processing_class 来把文本 + 图像 一起转成模型输入（多模态） #
        ##################################################################
        if any(img is not None for img in images):
            # ----- 多模态分支 -----
            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=images, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left", 
                add_special_tokens=False
            )
            #  这里的关键在于：prompt_inputs 必须包含
            #    - "input_ids" (batch_size, seq_len_text)
            #    - "attention_mask" (batch_size, seq_len_text)
            #    - "pixel_values"    (batch_size, 3, H, W) 或其他图像张量
            #    - "image_grid_thw"  (batch_size, G, T, H, W) 如果需要
            #  你要确认 processing_class.__call__ 的签名支持 text=…、images=…。
        else:
            # ----- 纯文本分支（保持向后兼容） ----
            prompt_inputs = self.processing_class(
                prompts_text, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left", 
                add_special_tokens=False
            )
            #  这里生成的 prompt_inputs 只有 "input_ids" 和 "attention_mask"。
        ########################################################################

        # 4. 把 tokenizer 输出的张量搬到正确设备（GPU/TPU/CPU）上
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # 此时 prompt_ids.shape == (batch_size, seq_len)，prompt_mask.shape == (batch_size, seq_len)
        
        # 5. 从 prompt_inputs 拿出图像对应的特征张量（如果有）
        pixel_values = prompt_inputs.get("pixel_values")       # 形状 (batch_size, 3, H, W) 或 None
        image_grid_thw = prompt_inputs.get("image_grid_thw")   # 形状 (batch_size, G, T, H, W) 或 None

        # 6. 如果有限制最大 prompt 长度，就截断最右侧
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # 7. 如果模型权重是新步数，就把模型放到 vLLM
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        #############################################################
        # 8. 收集（gather）所有进程上的 prompts 和 images （原始）  #
        #############################################################
        #    后面要把它们一起发给 vLLM 服务端做生成；只是收集最原始的 raw prompts & raw images（PIL或ndarray）
        all_prompts = gather_object(prompts)   # List[str]，长度=world_batch_size
        all_images = gather_object(images)     # List[PIL.Image or np.ndarray or None]，同长度

        #############################################################
        # 9. 只有主进程才真正调用 vLLM 服务做“生成”。               #
        #    因为无论文本还是图文，都要在同一端并行生成，然后广播到各进程。#
        #############################################################
        if self.accelerator.is_main_process:
            multimodal_inputs = []
            for prompt, image in zip(all_prompts, all_images):
                initial_prompts = CROP_SYSTEM_PROMPT.format(
                tool_descriptions=CROP_TOOL_DESCRIPTION,
                tool_example=CROP_TOOL_EXAMPLE
                ) + f'\nNow please help me to identify the coordinate of the following element : \n{prompt}'  # 添加系统提示  
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
                    # 纯文本输入
                    initial_message = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": initial_prompts},
                                ]
                            }
                        ]
                multimodal_inputs.append(initial_message)
            # 此时 multimodal_inputs 是一个长度等于 world_batch_size 的列表，
            # 其中每个元素形如 {"prompt": "...", "multi_modal_data": {"image": <PIL>} } 
            # 或者 {"prompt": "..."}。

            # 真正调用环境生成器（vLLM）：
            env_result = self.env.generate(
                prompts=multimodal_inputs,   # 多模态形式发出去
                llm=self.vllm_client,        # vLLM 客户端
                sampling_params=self.sampling_params,
            )
            # 返回值 env_result 中应该包含：
            #   - 'ids'：长度 = world_batch_size * num_generations，
            #       每个元素是某次生成的 token id list
            #   - 'messages'：长度 = world_batch_size * num_generations，
            #       每个元素是对应原始“prompt+image”后续生成的 message dict 列表
            #   - 'mask'：长度 = world_batch_size * num_generations，
            #       每个元素是生成长度对应的 attention mask（0/1列表）
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
        else:
            # 非主进程先占位，等待广播
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        ############################################################
        # 10. 广播主进程的 “completion_ids/messages/mask” 给各个子进程  #
        ############################################################
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        ############################################################
        # 11. 根据当前进程的 index，把全体 batch 切片到本地 batch 大小 #
        ############################################################
        #    world_batch_size = num_devices * local_batch_size
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        ) # len(prompts) 是每个进程的本地 batch 大小
        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        # 现在每个进程拿到自己的“部分生成结果”：
        #   completion_ids.shape == (local_batch_size, gen_len_unpadded_list)
        #   completion_messages: List(length=local_batch_size) 每个元素是 message dict
        #   completion_mask: List(length=local_batch_size) 每个元素是未 pad 的 mask list

        ##################################################
        # 12. 把每个 token_id list 转成 Tensor，再 pad 成统一长度 #
        ##################################################
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        # 现在 completion_ids.shape == (local_batch_size, max_gen_len)

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)
        # 现在 completion_mask.shape == (local_batch_size, max_gen_len)

        ##################################################
        # 13. 拼接 prompt_ids 与 completion_ids，得到完整上下文  #
        ##################################################
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # prompt_completion_ids.shape == (local_batch_size, prompt_len + gen_len)
        # attention_mask    == (local_batch_size, prompt_len + gen_len)

        logits_to_keep = completion_ids.size(1)
        # logits_to_keep = gen_len，告知后续只保留“本轮生成那部分”的 log_prob

        #####################################################
        # 14. 在 no_grad 下，用本地模型或参照模型算 per-token logp #
        #####################################################
        with torch.no_grad():
            # （1）如果 num_iterations>1，需要计算 old_per_token_logps 供 PPO ratio 计算
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,       # 多模态特征
                    image_grid_thw,     # 多模态网格
                    logits_to_keep
                )
            else:
                old_per_token_logps = None

            # （2）如果 beta>0 且给了一个 ref_model，需要对应计算参考模型 log‐probs
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep
                )
            else:
                # 关闭 adapter 时当基线模型
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep
                    )

        ###########################################################################
        # 15. 用 message dicts 作为 reward 函数的输入，开始对所有生成做打分    #
        ###########################################################################
        completions = completion_messages  # 仍然是 “message dict 列表”
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # 把 inputs[0] 中除了 “prompt” 和 “completion” 以外的其他字段都拿出来
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

            # 如果存在图像，就一并把 images 列表放入 reward_kwargs
            if any(img is not None for img in images):
                reward_kwargs["images"] = [example.get("image") for example in inputs]

            # reward_func 签名示例：reward_func(prompts=[str], completions=[msg_dict], images=[PIL], 其他字段=…)
            output_reward_func = reward_func(
                prompts=prompts,
                completions=completions,
                **reward_kwargs
            )

            # 把 None 转成 NaN，方便后续 nanmean/nansum
            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        ###############################################
        # 16. 如果某一行所有 reward 函数都返回 None，就警告 #
        ###############################################
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            if any(img is not None for img in images):
                row_reward_kwargs["image"] = images[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        ###############################################
        # 17. 在多卡情况下，对 rewards_per_func 做 all_gather #
        ###############################################
        rewards_per_func = gather(rewards_per_func)
        # 现在 rewards_per_func.shape == (world_batch_size, num_reward_funcs)

        ##################################
        # 18. 应用权重并对每条样本求和，得到总 reward #
        ##################################
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # rewards.shape == (world_batch_size,)

        ##########################################
        # 19. 计算分组内的平均奖励与优势 (advantage) #
        ##########################################
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        # mean_grouped_rewards.shape == (world_batch_size / num_generations,)

        # 把每组平均值 repeat_interleave 回到 (world_batch_size,) 
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards)

        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        #####################################################
        # 20. 把 advantages 按当前进程 index 切片回本地大小   #
        #####################################################
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        # 现在 advantages.shape == (local_batch_size,)

        #######################################
        # 21. 记录各项指标，供 TensorBoard/WandB 可视化 #
        #######################################
        mode = "eval" if self.control.should_evaluate else "train"

        # 记录平均生成长度
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # 对每个 reward function 记录 mean/std
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        ##########################################
        # 22. （可选）打印 / 上传样本到 WandB, 包含图像信息 #
        ##########################################
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            print("Logging completions...")
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            images_to_log = gather_object(images)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    # 打印带有“是否有图像”的示例
                    sample_prompt = str(prompts_to_log[0][-1]["content"])
                    if images_to_log[0] is not None:
                        sample_prompt += " [包含图像]"
                    print_prompt_completions_sample(
                        [sample_prompt],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                        "has_image": [img is not None for img in images_to_log],
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        #####################################################
        # 23. 返回所有供后续计算 loss 或保存的张量 / 数据       #
        #####################################################
        return {
            "prompt_ids": prompt_ids,             # (local_batch_size, prompt_len)
            "prompt_mask": prompt_mask,           # (local_batch_size, prompt_len)
            "completion_ids": completion_ids,     # (local_batch_size, gen_len)
            "completion_mask": completion_mask,   # (local_batch_size, gen_len)
            "old_per_token_logps": old_per_token_logps,   # (local_batch_size, gen_len) 或 None
            "ref_per_token_logps": ref_per_token_logps,   # (local_batch_size, gen_len) 或 None
            "advantages": advantages,             # (local_batch_size,)
            "pixel_values": pixel_values,         # (local_batch_size, 3, H, W) or None
            "image_grid_thw": image_grid_thw,     # (local_batch_size, G, T, H, W) or None
        }

    # def _generate_and_score_completions(
    #      self, inputs: dict[str, Union[torch.Tensor, Any]]   
    # ) -> dict[str, Union[torch.Tensor, Any]]:
    #     device = self.accelerator.device
    #     prompts = [x["prompt"] for x in inputs] # type: ignore
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
    #     prompt_inputs = self.processing_class(
    #         prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
    #     ) # type: ignore
    #     prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     if self.state.global_step != self._last_loaded_step:
    #         self._move_model_to_vllm()
    #         self._last_loaded_step = self.state.global_step

    #     # Gather the original prompts in message dict form, not the text form
    #     all_prompts = gather_object(prompts)
    #     if self.accelerator.is_main_process:
    #         env_result = self.env.generate(
    #             prompts=all_prompts,
    #             llm=self.vllm_client, # type: ignore
    #             sampling_params=self.sampling_params,
    #         )
    #         completion_ids = env_result['ids']
    #         completion_messages = env_result['messages']
    #         completion_mask = env_result['mask']

    #     else:
    #         completion_ids = [None] * len(all_prompts)
    #         completion_messages = [None] * len(all_prompts)
    #         completion_mask = [None] * len(all_prompts)

    #     completion_ids = broadcast_object_list(completion_ids, from_process=0)
    #     completion_messages = broadcast_object_list(completion_messages, from_process=0)
    #     completion_mask = broadcast_object_list(completion_mask, from_process=0)

    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )

    #     completion_ids = completion_ids[process_slice]
    #     completion_messages = completion_messages[process_slice]
    #     completion_mask = completion_mask[process_slice]

    #     # Pad + mask after per-sequence EOS tokens
    #     completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    #     completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

    #     completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
    #     completion_mask = pad(completion_mask, padding_value=0)

    #     prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)
        
    #     logits_to_keep = completion_ids.size(1)

    #     with torch.no_grad():
    #         # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
    #         # computation here, and use per_token_logps.detach() instead.
    #         if self.num_iterations > 1:
    #             old_per_token_logps = self._get_per_token_logps(
    #                 self.model, prompt_completion_ids, attention_mask, logits_to_keep
    #             )
    #         else:
    #             old_per_token_logps = None

    #         if self.beta == 0.0:
    #             ref_per_token_logps = None
    #         elif self.ref_model is not None:
    #             ref_per_token_logps = self._get_per_token_logps(
    #                 self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
    #             )
    #         else:
    #             with self.accelerator.unwrap_model(self.model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(
    #                     self.model, prompt_completion_ids, attention_mask, logits_to_keep
    #                 )

    #     # use message dicts for reward function inputs
    #     completions = completion_messages
    #     rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         # Repeat all input columns (but "prompt" and "completion") to match the number of generations
    #         keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
    #         reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
    #         output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore
            
    #         output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
    #         rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    #     # If all reward functions return None for a given row, issue a detailed warning
    #     if torch.isnan(rewards_per_func).all(dim=1).any():
    #         nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
    #         row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()} # type: ignore
    #         row_reward_kwargs["prompt"] = prompts[nan_row_idx]
    #         row_reward_kwargs["completion"] = completions[nan_row_idx] # type: ignore
    #         warnings.warn(
    #             f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
    #             "Please ensure that at least one reward function returns a valid reward."
    #         )


    #     rewards_per_func = gather(rewards_per_func)

    #     # Apply weights to each reward function's output and sum
    #     rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
    #     advantages = (rewards - mean_grouped_rewards)
        
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
    #     if self.scale_rewards:
    #         # Scale the rewards to be between 0 and 1
    #         advantages = advantages / (std_grouped_rewards + 1e-4)

    #     # Slice to keep only the local part of the data
    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     advantages = advantages[process_slice]

    #     # Log the metrics
    #     mode = "eval" if self.control.should_evaluate else "train"

    #     completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
    #     self._metrics[mode]["completion_length"].append(completion_length)

    #     # Calculate mean reward per function, but only for samples where the function was applied
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         reward_func_name = reward_func.__name__ # type: ignore  
    #         # Only calculate mean for samples where this reward function was applied (non-NaN values)
    #         mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
    #         self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
    #         std_rewards = nanstd(rewards_per_func[:, i]).item()
    #         self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
    #     self._metrics[mode]["reward"].append(rewards.mean().item())
    #     self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # type: ignore

    #     if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
    #         prompts_to_log = gather_object(prompts)
    #         completions_to_log = gather_object(completions)
    #         rewards_to_log = rewards.tolist()

    #         if self.accelerator.is_main_process:
    #             if is_rich_available():
    #                 print_prompt_completions_sample(
    #                     [str(prompts_to_log[0][-1]["content"])],
    #                     [completions_to_log[0]],
    #                     [rewards_to_log[0]],
    #                     self.state.global_step,
    #                 )
    #             if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
    #                 import pandas as pd

    #                 # For logging
    #                 table = {
    #                     "step": [str(self.state.global_step)] * len(rewards),
    #                     "prompt": prompts_to_log,
    #                     "completion": completions_to_log,
    #                     "reward": rewards.tolist(),
    #                 }
    #                 df = pd.DataFrame(table)
    #                 wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore

    #     return {
    #         "prompt_ids": prompt_ids,
    #         "prompt_mask": prompt_mask,
    #         "completion_ids": completion_ids,
    #         "completion_mask": completion_mask,
    #         "old_per_token_logps": old_per_token_logps,
    #         "ref_per_token_logps": ref_per_token_logps,
    #         "advantages": advantages,
    #     }