from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from datasets import Dataset, load_dataset
from transformers.utils import logging

logger = logging.get_logger("transformers")

TRAIN_EVAL_SPLIT_PERCENT: int = 20
PROCESSES_LIMIT: int = 1

class TrainDataBase(ABC):
    """
    """
    @abstractmethod
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len: int) -> None:
        """
        Args:
            dataset (str): Path to dataset
            val_set_size (int) : Size of validation set
            tokenizer (_type_): Tokenizer
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_set_size = val_set_size
        self.cutoff_len = cutoff_len
        self.train_data = None
        self.val_data = None

    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        """Loads dataset from file and prepares train_data for trainer."""
        pass


class TrainGPT4All(TrainDataBase):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        pass

    def tokenize_inputs(self, examples):
        max_length = self.cutoff_len
        input_ids = torch.full((len(examples["prompt"]), max_length), self.tokenizer.pad_token_id)
        # ignore bos
        newline_tokens = self.tokenizer("\n", return_tensors="pt")["input_ids"][0, 1:]

        out = {"labels": [], "attention_mask": []}
        for i, (prompt, response) in enumerate(zip(examples["prompt"], examples["response"])):
            input_tokens = self.tokenizer(prompt, truncation=True, max_length=max_length // 2, return_tensors="pt")["input_ids"].squeeze()
            if input_tokens.dim() == 0:
                input_tokens = input_tokens.unsqueeze(0)

            input_len = len(input_tokens)

            # plus one since we remove bos from response
            # but we subtract one since we want to add eos token
            remaining_tokens = max_length - input_len - len(newline_tokens) + 1
            # remove bos
            target_tokens = self.tokenizer(response, truncation=True, max_length=remaining_tokens, return_tensors="pt")["input_ids"].squeeze()[1:]

            input_ids[i, :input_len] = input_tokens
            # add newline between prompt and response
            newline_plus_inputs = input_len + len(newline_tokens)
            input_ids[i, input_len: newline_plus_inputs] = newline_tokens

            # add target tokens, remove bos
            input_ids[i, newline_plus_inputs: newline_plus_inputs + len(target_tokens)] = target_tokens
            # add eos token, enforce stopping if we don't truncate
            # we don't want long code to stop generating if truncated during training
            if newline_plus_inputs + len(target_tokens) < max_length:
                input_ids[i, newline_plus_inputs + len(target_tokens)] = self.tokenizer.eos_token_id

            labels = input_ids[i].clone()
            labels[: newline_plus_inputs] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            # to debug this, can set all values == -100 to the pad token, then assert that tokenizer.decode(labels, skip_special_tokens=True).strip() == response

            attention_mask = input_ids[i].ne(self.tokenizer.pad_token_id).int()

            out["labels"].append(labels)
            out["attention_mask"].append(attention_mask)

        out["input_ids"] = input_ids

        out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}

        return out

    def prepare_data(self, **kwargs) -> None:
        dataset = load_dataset("json", data_files=self.dataset)

        self.val_data = None
        if self.val_set_size > 0:
            dataset = dataset["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42  # ! Seed = 42 (?)
            )
            train_dataset, val_dataset = dataset["train"], dataset["test"]

            # tokenize inputs and return labels and attention mask
            val_dataset = val_dataset.map(
                lambda ele: self.tokenize_inputs(ele),
                batched=True,
                remove_columns=["source", "prompt"],
            )
            self.val_data = val_dataset.with_format("torch")
        else:
            train_dataset = dataset["train"]

        train_dataset = train_dataset.map(
            lambda ele: self.tokenize_inputs(ele),
            batched=True,
            remove_columns=["source", "prompt"],
        )
        self.train_data = train_dataset.with_format("torch")


class TrainSAD(TrainDataBase):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        data = load_dataset("json", data_files=self.dataset)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(test_size=self.val_set_size, shuffle=True, seed=42)
            self.train_data = train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        else:
            self.train_data = data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = None

    # Auxiliary methods
    def generate_prompt(self, data_point, **kwargs):
        return make_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"]
        )

    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)


class TrainLAIKA(TrainDataBase):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }


    def encode(self, examples, tokenizer, text_column_name):
        return tokenizer(examples[text_column_name])


    def prepare_dataset(self, path_to_file, tokenizer):
        # disable_progress_bar()
        data_files = dict()
        data_files["train"] = path_to_file

        dataset: Dataset = load_dataset(
            "text", data_files=data_files, keep_linebreaks=True
        )

        dataset["validation"] = load_dataset(
            "text",
            data_files=data_files,
            split=f"train[:{TRAIN_EVAL_SPLIT_PERCENT}%]",
        )

        dataset["train"] = load_dataset(
            "text",
            data_files=data_files,
            split=f"train[{TRAIN_EVAL_SPLIT_PERCENT}%:]",
        )

        column_names = dataset["train"].column_names

        text_column_name: str = "text" if "text" in column_names else column_names[0]

        dataset = dataset.map(
            self.encode,
            batched=True,
            num_proc=PROCESSES_LIMIT,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
            fn_kwargs={"tokenizer": tokenizer,
                    "text_column_name": text_column_name},
        )

        block_size: int = tokenizer.model_max_length
        if block_size > 1024:
            # logger.warning(
            #     f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            #     "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            # )
            block_size = 1024

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        lm_datasets: Dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=PROCESSES_LIMIT,
            desc=f"Grouping texts in chunks of {block_size}",
            fn_kwargs={"block_size": block_size},
        )

        train_dataset: Dataset = lm_datasets["train"]

        eval_dataset: Dataset = lm_datasets["validation"]

        return train_dataset, eval_dataset

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        # data = load_dataset("json", data_files=self.dataset)

        # data = load_dataset(
        #     "text", data_files=self.dataset, keep_linebreaks=True
        # )

        self.train_data, self.val_data = self.prepare_dataset(
            self.dataset, self.tokenizer
        )

        # if self.val_set_size > 0:
        #     train_val = data["train"].train_test_split(
        #         test_size=self.val_set_size, shuffle=True, seed=42)
        #     self.train_data = train_val["train"].shuffle().map(
        #         lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        #     self.val_data = train_val["validation"].shuffle().map(
        #         lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        # else:
        #     self.train_data = data["train"].shuffle().map(
        #         lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        #     self.val_data = None

    # # Auxiliary methods
    # def generate_prompt(self, data_point, **kwargs):
    #     return make_prompt(
    #         data_point["instruction"],
    #         data_point["input"],
    #         data_point["output"]
    #     )

    # def generate_and_tokenize_prompt(self, data_point, **kwargs):
    #     prompt = self.generate_prompt(data_point, **kwargs)
    #     return self.tokenize(prompt, **kwargs)



def make_prompt(instruction, input_, output=""):
    return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}\n{6}".format(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        "### Instruction:",
        instruction,
        "### Input:",
        input_,
        "### Response:",
        output
    )


def load_data(config, tokenizer):
    if config.data_type == "alpaca":
        data = TrainSAD(
            config.dataset,
            config.val_set_size,
            tokenizer,
            config.cutoff_len)

    elif config.data_type == "gpt4all":
        data = TrainGPT4All(
            config.dataset,
            config.val_set_size,
            tokenizer,
            config.cutoff_len)

    elif config.data_type == "LAIKA":
        data = TrainLAIKA(
            config.dataset,
            config.val_set_size,
            tokenizer,
            config.cutoff_len)
    else:
        raise ValueError(f"Invalid data name: {config.data_type}")

    data.prepare_data(use_eos_token=config.use_eos_token)
    return data


DATA_TYPES = [
    "alpaca",
    "gpt4all",
    "LAIKA",
]
