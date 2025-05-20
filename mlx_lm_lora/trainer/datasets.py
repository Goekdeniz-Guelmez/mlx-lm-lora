from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
import types
import json

from transformers import PreTrainedTokenizer


class GRPODataset:
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        system_key: str = "system",
        type_key: str = "type",
        use_chat_template: bool = False,
        use_prompt: bool = False
    ):
        self._data = []
        for item in data:
            # Get prompt with fallbacks for different key names
            for key in [prompt_key, "instruction", "question"]:
                if key in item:
                    prompt_str = str(item[key])
                    break
            else:
                raise ValueError(f"No prompt field found in data. Expected one of: {prompt_key}, 'instruction', 'question'")
                
            # Get answer with fallbacks for different key names
            for key in [answer_key, "output", "response", "completion"]:
                if key in item:
                    answer_str = str(item[key])
                    break
            else:
                raise ValueError(f"No answer field found in data. Expected one of: {answer_key}, 'output', 'response', 'completion'")
                
            # Get optional type info
            type_info = item.get(type_key, None)
            
            if use_chat_template:
                default_system_str = "You are a helpful assistant."
                system_str = item.get(system_key, default_system_str)
                prompt_tokens = tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': system_str},
                        {'role': 'user', 'content': prompt_str}
                    ],
                    add_generation_prompt=True
                )
                answer_tokens = tokenizer.encode(answer_str)
            else:
                # If system prompt is provided, prepend it to the prompt
                if system_key in item:
                    system_str = str(item[system_key])
                    if use_prompt:
                        prompt_tokens = tokenizer.encode(f"{system_str}\n\n{prompt_str}")
                    else:
                        # For poker agent, just use the regular format with system as a separate field
                        prompt_tokens = tokenizer.encode(prompt_str)
                else:
                    if use_prompt:
                        prompt_tokens = tokenizer.encode(f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. User: {prompt_str} Assistant: ")
                    else:
                        prompt_tokens = tokenizer.encode(prompt_str)
                
                answer_tokens = tokenizer.encode(answer_str)
                
            self._data.append((prompt_tokens, answer_tokens, prompt_str, answer_str, type_info))

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], str, str]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def process(self, d):
        return d


class DPODataset:
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        system_key: str = "system",
    ):
        self._chosen_data = []
        self._rejected_data = []

        for d in data:
            messages = (
                [{"role": "system", "content": d[system_key]}]
                if system_key and system_key in d
                else []
            )
            messages.append({"role": "user", "content": d[prompt_key]})

            base_messages = messages.copy()
            chosen_messages = base_messages + [
                {"role": "assistant", "content": d[chosen_key]}
            ]
            rejected_messages = base_messages + [
                {"role": "assistant", "content": d[rejected_key]}
            ]

            self._chosen_data.append(tokenizer.apply_chat_template(chosen_messages))
            self._rejected_data.append(tokenizer.apply_chat_template(rejected_messages))

    def __getitem__(self, idx: int):
        return {"chosen": self._chosen_data[idx], "rejected": self._rejected_data[idx]}

    def __len__(self):
        return len(self._chosen_data)

    def process(self, d):
        return d
    

class ORPODataset:
    def __init__(
        self,
        data: List[Dict[str, Union[str, Dict, List]]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        preference_score_key: str = "preference_score",
        system_key: str = None,
    ):
        self._chosen_data = []
        self._rejected_data = []
        self._scores = []

        for d in data:
            prompt_content = d.get(prompt_key, d.get("question", ""))

            if system_key and system_key in d:
                base_messages = [{"role": "system", "content": d[system_key]}]
                chosen_messages = base_messages + [
                    {"role": "user", "content": prompt_content}
                ]
                rejected_messages = base_messages + [
                    {"role": "user", "content": prompt_content}
                ]

                if isinstance(d[chosen_key], str):
                    chosen_messages.append(
                        {"role": "assistant", "content": d[chosen_key]}
                    )
                elif isinstance(d[chosen_key], dict):
                    if "messages" in d[chosen_key]:
                        chosen_messages.extend(d[chosen_key]["messages"])
                    else:
                        chosen_messages.append(
                            {
                                "role": "assistant",
                                "content": d[chosen_key].get("content", ""),
                            }
                        )
                elif isinstance(d[chosen_key], list):
                    chosen_messages.extend(d[chosen_key])

                if isinstance(d[rejected_key], str):
                    rejected_messages.append(
                        {"role": "assistant", "content": d[rejected_key]}
                    )
                elif isinstance(d[rejected_key], dict):
                    if "messages" in d[rejected_key]:
                        rejected_messages.extend(d[rejected_key]["messages"])
                    else:
                        rejected_messages.append(
                            {
                                "role": "assistant",
                                "content": d[rejected_key].get("content", ""),
                            }
                        )
                elif isinstance(d[rejected_key], list):
                    rejected_messages.extend(d[rejected_key])

                chosen_text = tokenizer.apply_chat_template(chosen_messages)
                rejected_text = tokenizer.apply_chat_template(rejected_messages)

            else:
                chosen_content = self._extract_content(d[chosen_key])
                rejected_content = self._extract_content(d[rejected_key])

                chosen_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt_content},
                        {"role": "assistant", "content": chosen_content},
                    ]
                )
                rejected_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt_content},
                        {"role": "assistant", "content": rejected_content},
                    ]
                )

            self._chosen_data.append(chosen_text)
            self._rejected_data.append(rejected_text)

            if preference_score_key in d:
                self._scores.append(float(d[preference_score_key]))
            else:
                self._scores.append(1.0)

    def _extract_content(self, data):
        """Helper method to extract content from various data formats."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if "messages" in data:
                last_message = data["messages"][-1]
                return last_message.get("content", last_message.get("messages", ""))
            return data.get("content", "")
        elif isinstance(data, list):
            last_message = data[-1]
            if isinstance(last_message, dict):
                if "content" in last_message:
                    return last_message["content"]
                elif "messages" in last_message:
                    return last_message["messages"]
            return last_message if isinstance(last_message, str) else ""
        return ""

    def __len__(self):
        return len(self._chosen_data)

    def process(self, d):
        return d

    def __getitem__(self, idx: int):
        return {
            "chosen": self._chosen_data[idx],
            "rejected": self._rejected_data[idx],
            "preference_score": self._scores[idx],
        }


class TextDataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = data
        self.tokenizer = tokenizer
        self.text_key = text_key

    def process(self, d):
        d = self.tokenizer.encode(d[self.text_key])
        if d[-1] != self.tokenizer.eos_token_id:
            d.append(self.tokenizer.eos_token_id)
        return d

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ChatDataset:
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
        mask_prompt: bool = False,
    ):
        self._data = data
        self.chat_key = chat_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d):
        messages = d[self.chat_key]
        tools = d.get("tools", None)
        tokens = self.tokenizer.apply_chat_template(messages, tools=tools)
        if self.mask_prompt:
            messages = messages[:-1]
            offset = len(self.tokenizer.apply_chat_template(messages, tools=tools))
            return (tokens, offset)
        else:
            return tokens

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class CompletionsDataset:
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str,
        completion_key: str,
        mask_prompt: bool,
    ):
        self._data = data
        self.prompt_key = prompt_key
        self.completion_key = completion_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d):
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": d[self.prompt_key]},
                {"role": "assistant", "content": d[self.completion_key]},
            ],
        )
        if self.mask_prompt:
            offset = len(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": d[self.prompt_key]}]
                )
            )
            return (tokens, offset)

        return tokens

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ConcatenatedDataset:
    def __init__(self, data: List[Any]):
        self._data = data
        self._len = sum(len(d) for d in self._data)

    def __getitem__(self, idx: int):
        for data_idx, data in enumerate(self._data):
            j = idx - len(data)
            if j < 0:
                break
            idx = j
        datum = data[idx]
        datum["_dataset"] = data_idx
        return datum

    def process(self, d):
        return self._data[d["_dataset"]].process(d)

    def __len__(self):
        return self._len


class CacheDataset:
    def __init__(self, data: Any):
        self._data = data
        self._proc_data = [None] * len(data)

    def itemlen(self, idx: int):
        return len(self._data[idx])

    def __getitem__(self, idx: int):
        if self._proc_data[idx] is None:
            self._proc_data[idx] = self._data.process(self._data[idx])
        return self._proc_data[idx]

    def __len__(self):
        return len(self._data)


def create_dataset(
    data,
    tokenizer: PreTrainedTokenizer,
    config,
):
    mask_prompt = getattr(config, "mask_prompt", False)
    train_mode = getattr(config, "train_mode", "sft")

    text_feature = getattr(config, "text_feature", "text")
    chat_feature = getattr(config, "chat_feature", "messages")
    prompt_feature = getattr(config, "prompt_feature", "prompt")
    completion_feature = getattr(config, "completion_feature", "completion")

    # For ORPO and DPO
    system_feature = getattr(config, "system_feature", "system")
    chosen_feature = getattr(config, "chosen_feature", "chosen")
    rejected_feature = getattr(config, "rejected_feature", "rejected")
    preference_score_feature = getattr(config, "preference_score_feature", "preference_score")

    # For GRPO
    use_prompt = getattr(config, "use_prompt", False)
    type_feature = getattr(config, "type_feature", "type")
    answer_feature = getattr(config, "answer_feature", "answer")
    use_chat_template = getattr(config, "use_chat_template", False)

    # Make sure we have at least one entry in the dataset
    if not data or len(data) == 0:
        return []
    
    # Get the first sample to check the format
    sample = data[0]
    
    # For GRPO, just check if necessary fields exist in any form
    if train_mode == "grpo":
        # Look for prompt field with alternative names if needed
        prompt_exists = any(field in sample for field in [prompt_feature, "instruction", "question"])
        # Look for answer field with alternative names if needed
        answer_exists = any(field in sample for field in [answer_feature, "output", "response", "completion"])
        
        if prompt_exists and answer_exists:
            return GRPODataset(
                data=data,
                tokenizer=tokenizer,
                prompt_key=prompt_feature,
                answer_key=answer_feature,
                system_key=system_feature,
                type_key=type_feature,
                use_chat_template=use_chat_template,
                use_prompt=use_prompt
            )
        else:
            raise ValueError(f"Unsupported data format for GRPO training. Required fields: {prompt_feature} and {answer_feature}. Found: {list(sample.keys())}")
    elif train_mode == "orpo":
        if chosen_feature in sample and rejected_feature in sample:
            return ORPODataset(
                data=data,
                tokenizer=tokenizer,
                system_key=system_feature,
                prompt_key=prompt_feature,
                chosen_key=chosen_feature,
                rejected_key=rejected_feature,
                preference_score_key=preference_score_feature
            )
        else:
            raise ValueError("Unsupported data format for ORPO training.")
    elif train_mode == "dpo" or train_mode == "cpo":
        if chosen_feature in sample and rejected_feature in sample:
            return DPODataset(
                data=data,
                tokenizer=tokenizer,
                prompt_key=prompt_feature,
                system_key=system_feature,
                chosen_key=chosen_feature,
                rejected_key=rejected_feature
                )
        else:
            raise ValueError("Unsupported data format for DPO training.")
    elif train_mode == "sft":
        if prompt_feature in sample and completion_feature in sample:
            return CompletionsDataset(
                data, tokenizer, prompt_feature, completion_feature, mask_prompt
            )
        elif chat_feature in sample:
            return ChatDataset(
                data, tokenizer, chat_key=chat_feature, mask_prompt=mask_prompt
            )
        elif text_feature in sample:
            if mask_prompt:
                raise ValueError("Prompt masking not supported for text dataset.")
            return TextDataset(data, tokenizer, text_key=text_feature)
        else:
            raise ValueError("Unsupported data format for SFT training.")
    else:
        raise ValueError(f"Unsupported training mode: {train_mode}")


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config,
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(data, tokenizer, config)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config,
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            (
                create_dataset(dataset[n], tokenizer, config)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    import datasets

    def create_hf_dataset(dataset_name, config, split, hf_config):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_config,
        )
        return create_dataset(ds, tokenizer, config)

    dataset_collection = args.hf_dataset
    if isinstance(dataset_collection, dict):
        dataset_collection = [dataset_collection]

    collection = []
    for ds in dataset_collection:
        ds_path = ds["path"]
        print(f"Loading Hugging Face dataset {ds_path}.")
        ds["mask_prompt"] = getattr(args, "mask_prompt", False)
        config = types.SimpleNamespace(**ds)
        hf_config = ds.get("config", {})
        if args.train:
            train_split = ds.get("train_split", "train[:80%]")
            valid_split = ds.get("valid_split", "train[-10%:]")
            train = create_hf_dataset(
                ds_path,
                config,
                train_split,
                hf_config,
            )
            valid = create_hf_dataset(
                ds_path,
                config,
                valid_split,
                hf_config,
            )
        else:
            train, valid = [], []

        if args.test:
            test_split = ds.get("test_split")
            test = create_hf_dataset(
                ds_path,
                config,
                test_split,
                hf_config,
            )
        else:
            test = []

        collection.append((train, valid, test))

    if len(collection) == 1:
        return collection[0]

    # Otherwise concatenate them
    return tuple(map(ConcatenatedDataset, zip(*collection)))


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(data_path, tokenizer, args)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args.data, tokenizer, args)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
