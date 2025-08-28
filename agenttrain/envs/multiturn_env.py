from abc import abstractmethod
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
from typing import List, Dict, Sequence, Any, Union, Tuple
import base64
import io
from PIL import Image

from datasets import Dataset
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from agenttrain.inference.vllm_client import VLLMClient

from agenttrain.envs.environment import Environment
from agenttrain.utils import format_dataset
from agenttrain.utils.data_utils import sanitize_dialogs

class ChatOutput(BaseModel):
    token_ids: List[int]
    text: str

class ChatResponseItem(BaseModel):
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[ChatOutput]

class ChatResponse(BaseModel):
    responses: List[ChatResponseItem]

def dict_to_chat_response(data: Dict[str, Any]) -> ChatResponse:
    """
    Recursively convert a dictionary to a ChatResponse object
    """
    # First, convert all outputs to ChatOutput objects
    if "responses" in data:
        for i, response_item in enumerate(data["responses"]):
            if "outputs" in response_item:
                data["responses"][i]["outputs"] = [
                    ChatOutput(**output) for output in response_item["outputs"]
                ]
        
        # Then convert all response items to ChatResponseItem objects
        data["responses"] = [ChatResponseItem(**item) for item in data["responses"]]
    
    # Finally, convert the entire dict to a ChatResponse object
    return ChatResponse(**data)

class MultiTurnEnv(Environment):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 16,
                 max_steps: int = 10,
                 sleep_time: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.few_shot = few_shot
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.max_steps = max_steps

    @abstractmethod
    def is_completed(self, messages: List[dict[str, Union[str, List[dict]]]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self, messages: List[dict[str, Union[str, List[dict]]]], **kwargs: Any) -> Dict[str, str]:
        pass

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM | VLLMClient,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]

        if isinstance(llm, VLLMClient):
            llm_responses = []
            n = len(messages_to_step)
            print(f"Number of messages to process: {n}")
            i = 0

            while i < n:
                # 取 [i, i+chunk_size) 这一批（最后一批如果不足 chunk_size 也会取到末尾）
                batch = messages_to_step[i : i + n]
                # batch = messages_to_step[i : i + 1]
                # print(f"Sample message: {batch[0]}")
                # print(f"max_tokens: {sampling_params.max_tokens}")
                resp = llm.chat(
                    batch,
                    n=1,
                    repetition_penalty=sampling_params.repetition_penalty,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    top_k=sampling_params.top_k,
                    min_p=sampling_params.min_p,              # type: ignore
                    max_tokens=sampling_params.max_tokens,    # type: ignore
                    stop=sampling_params.stop,                # type: ignore
                    include_stop_str_in_output=sampling_params.include_stop_str_in_output,
                    skip_special_tokens=sampling_params.skip_special_tokens,
                    spaces_between_special_tokens=sampling_params.spaces_between_special_tokens
                )  # type: ignore
                # print(f"Response for batch {i // n} finished, length: {len(resp)}")
                sub_resps = dict_to_chat_response(resp).responses
                llm_responses.extend(sub_resps)

                i += n
                # i += 1
        else:
            llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=True) # type: ignore
        
        def update_state(j, llm_response):
            """
            Update three things in the state:
            1. messages: append the assistant response
            2. all_prompts: include the prompt token ids and the assistant response text from all turns
            3. images: append the image from the tools if it exists
            """
            # sleep for 0-1 seconds to avoid rate limiting
            time.sleep(self.sleep_time * random.random())
            state = deepcopy(states[j])
            
            # Avoid image padding in the response
            # OtherWise there is some chance that the ERROR: 
            # num_image_tokens = image_grid_thw[index].prod() // merge_length IndexError: will happen
            clean_text = llm_response.outputs[0].text.replace('<|image_pad|>', '')
            state["messages"].append({"role": "assistant", "content": [{'type': 'text', 'text': clean_text}]})
        
            # Finish or execute the tools
            current_id_length = len(llm_response.prompt_token_ids) + len(llm_response.outputs[0].token_ids)
            # print(f"Current ID length: {current_id_length}, Max tokens: {sampling_params.max_tokens}")
            if self.is_completed(state["messages"]) or current_id_length > sampling_params.max_tokens - 1:
                # print(f"Marking state {j} as completed")
                state["completed"] = True
                state['all_prompts'] = llm_response.prompt + clean_text + '<|im_end|>' # update all_prompts
            else:
                self.env_response(state["messages"], state["images"], 
                                  state["images_offset"], state['tool_used']) # call tools and add environment response

            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
            ))

        for j, state in results:
            states[j] = state

        return states

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM | VLLMClient,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        
        """
        Generate responses for multiple prompts using the LLM and sampling parameters.
        Args:
            prompts: List of prompts, each a list of message dicts
            llm: LLM or VLLMClient instance for generating responses
            sampling_params: Sampling parameters for the generation
            **kwargs: Additional arguments (not used here)
        Returns:
            A dictionary containing:
            - all_prompts: List of all prompts generated by the LLM
            - images: List of images generated by the tools, if any
        """
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        def bs64_image(messages) -> str:
            image_entry = next(item for item in messages[0]["content"] if item["type"] == "image_url")
            data_uri = image_entry["image_url"]["url"]
            bs64_str = data_uri.split(",", 1)[1]
            image_bytes = base64.b64decode(bs64_str)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return img
        
        # initialize state variables
        all_completed = False
        states = []
        for m in prompts:
            img = bs64_image(m)
            state = {
                'tool_used': [],
                "messages": m,
                "all_prompts": "",
                "completed": False,
                "images": [img],
                "images_offset": [(0,0)],  # Store additional image info if needed
            }
            states.append(state)
        
        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)
            # print(f"All completed: {all_completed}, Remaining: {sum(not s['completed'] for s in states)}")
            
        all_prompts = [s["all_prompts"] for s in states]
        all_images = [s["images"] for s in states] # list[list[Image.Image]] 
        all_messages = [s["messages"] for s in states]
        all_images_offset = [s["images_offset"] for s in states] # list[list[Tuple[int, int]]] 
        all_tool_used = [s["tool_used"] for s in states] # list[list[str]]
        
        output = {
            "all_prompts": all_prompts,
            "images": all_images,
            "all_messages": all_messages,
            "images_offset": all_images_offset,
            "all_tool_used": all_tool_used
        }
        return output

    def step_api(self, 
             client: Any,
             model: str,
             messages: List[Dict[str, str]],
             sampling_args: Dict[str, Any] = {},
             **kwargs: Any) -> Tuple[List[Dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API
        
        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = deepcopy(messages)
        
        try:            
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
                extra_body=sampling_args
            )
            
            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant", 
                "content": response.choices[0].message.content
            }
            messages_copy.append(assistant_msg)
            
            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(messages_copy)
                messages_copy.append(env_msg)
            
            return messages_copy, rollout_is_completed
            
        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True
    
    def eval_api(self, 
                client: Any,
                model: str,
                max_concurrent: int = 32,
                timeout: int = 60,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any):
        
        eval_sampling_args = deepcopy(self.sampling_args)
        eval_sampling_args.update(sampling_args)
        """
        Evaluate model using OpenAI API with proper concurrency.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Tuple of (eval_dataset, rewards)
        """
        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore
            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)
            
            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")
            
            eval_dataset = self.eval_dataset
            
            async def process_example(example, semaphore):
                async with semaphore:
                    # Initialize conversation with system prompt and few-shot examples
                    prompt = example["prompt"]
                    messages = deepcopy(example["prompt"])
                    answer = example["answer"]
                    
                    # Save the length of initial messages to extract just the interaction part later
                    initial_length = len(messages)

                    # Run the conversation loop until completion or max steps
                    for _ in range(self.max_steps):  # Safety limit on conversation turns
                        try:
                            # Run step_api to get model and environment response
                            # Note: step_api now returns a tuple (messages, is_completed)
                            step_result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.step_api(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    sampling_args=eval_sampling_args
                                )
                            )
                            
                            # Unpack the step_api result
                            messages, is_completed = step_result
                            
                            # If the rollout is completed, break the loop
                            if is_completed:
                                break
                            
                        except Exception as e:
                            # print(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                            break
                    
                    # Extract only the interaction part (not system/few-shot)
                    completions = messages[initial_length:]
                    
                    return {
                        "prompt": prompt,
                        "completions": completions,
                        "task": example["task"],
                        "answer": answer
                    }
            
            async def run_all_examples():
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)
                
                # Process all examples concurrently
                tasks = [process_example(example, semaphore) for example in eval_dataset]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples"
                )
                
                return results
            
            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()
            
            # Calculate rewards
            results_prompt = [result["prompt"] for result in results]
            results_answer = [result["answer"] for result in results]
            results_task = [result["task"] for result in results]
            results_completions = [result["completions"] for result in results]
            results = {"prompt": results_prompt, "answer": results_answer, "completions": results_completions, "task": results_task}
            
            reward_funcs = self.get_reward_funcs()
            rewards = {}
            
            for reward_func in reward_funcs:
                func_rewards = reward_func(**results) # type: ignore
                func_rewards = [fr for fr in func_rewards if fr is not None]
                func_reward_avg = sum(func_rewards) / max(1, len(func_rewards))
                func_name = reward_func.__name__ # type: ignore
                # print(f"{func_name}: {func_reward_avg}")
                rewards[func_name] = func_reward_avg
            
            return rewards
            
        # Run the evaluation function
        return run_evaluation()
    

    