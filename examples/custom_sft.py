import mlx.optimizers as optim
from datasets import concatenate_datasets, load_dataset
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.utils import load

from mlx_lm_lora.trainer.datasets import CacheDataset, TextDataset
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft

system_prompt = """You are **J.O.S.I.E.**, an advanced super-intelligent AI Assistant created by a 25 year old machine learning researcher named **Gökdeniz Gülmez**."""
model_name = "mlx-community/Josiefied-Qwen3-0.6B-abliterated-v1-4bit"
new_model_name = "Josiefied-Qwen3-14B-abliterated-v3"
adapter_path = "path/to/adapters"
max_seq_length = 512

dataset_names = ["Goekdeniz-Guelmez/Wizzard-smol"]
dataset_samples = None


model, tokenizer = load(model_name)


opt = optim.AdamW(learning_rate=1e-5)


def format_prompts_func(sample):
    this_conversation = sample["conversations"]

    if isinstance(this_conversation, list):
        conversation = []
        conversation.append({"role": "system", "content": system_prompt})
        for turn in this_conversation:
            if turn["from"] == "human":
                conversation.append({"role": "user", "content": turn["value"]})
            elif turn["from"] == "gpt":
                conversation.append({"role": "assistant", "content": turn["value"]})

    sample["text"] = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=False,
        enable_thinking=False,
        tokenize=False,
    )
    return sample


datasets = [load_dataset(name)["train"] for name in dataset_names]
combined_dataset = concatenate_datasets(datasets)

if dataset_samples is not None:
    combined_dataset = combined_dataset.select(range(dataset_samples))

full_dataset = combined_dataset.map(
    format_prompts_func,
)
train_dataset, valid_dataset = full_dataset.train_test_split(
    test_size=0.01, seed=42
).values()


train_set = TextDataset(train_dataset, tokenizer, text_key="text")
valid_set = TextDataset(train_dataset, tokenizer, text_key="text")


train_sft(
    model=model,
    args=SFTTrainingArgs(
        batch_size=1,
        iters=100,
        val_batches=1,
        steps_per_report=20,
        steps_per_eval=50,
        steps_per_save=50,
        adapter_file="path/to/adapter/file",
        max_seq_length=512,
        grad_checkpoint=True,
    ),
    optimizer=opt,
    train_dataset=CacheDataset(train_set),
    val_dataset=CacheDataset(valid_set),
    training_callback=TrainingCallback(),
)
