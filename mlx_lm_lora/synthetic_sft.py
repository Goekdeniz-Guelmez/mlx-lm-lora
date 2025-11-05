import argparse
import json
import random
import re
import sys

from datasets import load_dataset
from mlx_lm.generate import generate
from mlx_lm.utils import load


def load_model(model_path):
    """Load MLX model and tokenizer"""
    try:
        model, tokenizer = load(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise


def extract_thinking_tags(text):
    """Extract thinking/reasoning content from text within <think> tags"""
    # Pattern to match <think>...</think> tags (case insensitive, multiline)
    thinking_pattern = r"<think>(.*?)</think>"
    matches = re.findall(thinking_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Join multiple thinking blocks if present
        reasoning = "\n\n".join(match.strip() for match in matches)
        # Remove thinking tags from original text
        cleaned_text = re.sub(
            thinking_pattern, "", text, flags=re.DOTALL | re.IGNORECASE
        )
        # Clean up extra whitespace
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text.strip())
        return cleaned_text, reasoning

    return text, None


def load_prompts_from_hf(dataset_name, split="train", max_samples=None):
    """Load prompts from Hugging Face dataset"""
    try:
        print(f"📥 Loading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        prompts = []
        for item in dataset:
            if "prompt" in item:
                prompts.append(item["prompt"])
            elif "messages" in item:
                # Extract user message from messages format
                messages = item["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    user_msg = next(
                        (msg["content"] for msg in messages if msg["role"] == "user"),
                        None,
                    )
                    if user_msg:
                        prompts.append(user_msg)
            else:
                print(
                    f"⚠️  Warning: Item missing 'prompt' or 'messages' field: {item.keys()}"
                )

        print(f"✅ Loaded {len(prompts)} prompts from Hugging Face dataset")
        return prompts

    except Exception as e:
        print(f"❌ Error loading from Hugging Face: {e}")
        return []


def load_prompts_from_jsonl(file_path, max_samples=None):
    """Load prompts from local JSONL file"""
    try:
        print(f"📥 Loading prompts from local file: {file_path}")
        prompts = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(prompts) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    if "prompt" in item:
                        prompts.append(item["prompt"])
                    elif "messages" in item:
                        # Extract user message from messages format
                        messages = item["messages"]
                        if isinstance(messages, list) and len(messages) > 0:
                            user_msg = next(
                                (
                                    msg["content"]
                                    for msg in messages
                                    if msg["role"] == "user"
                                ),
                                None,
                            )
                            if user_msg:
                                prompts.append(user_msg)
                    else:
                        print(
                            f"⚠️  Warning: Line {line_num} missing 'prompt' or 'messages' field"
                        )

                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                    continue

        print(f"✅ Loaded {len(prompts)} prompts from local file")
        return prompts

    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading from file: {e}")
        return []


def generate_user_message(
    model,
    tokenizer,
    args,
    prompt_or_topic,
    conversation_history=None,
    is_from_dataset=False,
):
    """Generate a user message for the given prompt/topic"""
    if is_from_dataset and not conversation_history:
        # Use the prompt directly from dataset for first turn
        return prompt_or_topic

    if not conversation_history:
        system_prompt = f"You are to adapt the role of a human user. {args.user_role}."
        user_prompt = f"You are struggling with this topic: '{prompt_or_topic}'. Ask a specific question or describe a concrete issue you're facing. Only return the question."
    else:
        system_prompt = f"You are a user continuing a conversation. {args.user_role}"
        last_assistant_msg = conversation_history[-1]["content"]
        user_prompt = f"The assistant just said:\n\n{last_assistant_msg}\n\nNow ask a follow-up question or respond naturally to continue the conversation about {prompt_or_topic}. Still only respond with the User Turn Question."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=args.max_tokens,
        )

        return response.strip()
    except Exception as e:
        raise SystemError(f"Error generating user message: {e}")


def generate_assistant_message(
    model, tokenizer, args, user_message, topic, is_final_turn=False
):
    """Generate an assistant response to the user message"""
    system_prompt = (
        args.system_prompt
        or f"You are {args.assistant_name}, {args.assistant_role}. Provide helpful, accurate, and detailed responses about MLX and machine learning topics."
    )

    conclusion_instruction = (
        " Please provide a concise summary or conclusion to wrap up this topic."
        if is_final_turn
        else ""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_message}{conclusion_instruction}"},
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=args.max_tokens,
        )

        return response.strip()
    except Exception as e:
        raise SystemError(f"Error generating assistant message: {e}")


def generate_conversation(
    model, tokenizer, args, prompt_or_topic, is_from_dataset=False
):
    """Generate a complete conversation for the given prompt/topic"""
    conversation = []
    num_turns = random.randint(1, args.max_turns)

    topic_display = (
        prompt_or_topic[:50] + "..." if len(prompt_or_topic) > 50 else prompt_or_topic
    )
    print(f"  Generating {num_turns} turns for: {topic_display}")

    for turn in range(num_turns):
        # Generate user message
        user_message = generate_user_message(
            model,
            tokenizer,
            args,
            prompt_or_topic,
            conversation_history=conversation if turn > 0 else None,
            is_from_dataset=is_from_dataset,
        )
        user_message, _ = extract_thinking_tags(user_message)
        conversation.append({"role": "user", "content": user_message})

        # Generate assistant response
        is_final = turn == num_turns - 1
        assistant_message = generate_assistant_message(
            model,
            tokenizer,
            args,
            user_message,
            prompt_or_topic,
            is_final_turn=is_final,
        )

        # Extract thinking tags if present
        cleaned_message, reasoning = extract_thinking_tags(assistant_message)

        # Create assistant message entry
        assistant_entry = {"role": "assistant", "content": assistant_message}
        if reasoning:
            assistant_entry["reasoning"] = reasoning
            assistant_entry["answer"] = cleaned_message

        conversation.append(assistant_entry)

        if args.dry_run:
            print(f"    Turn {turn + 1}:")
            print(f"    User: {user_message[:100]}...")
            print(f"    Assistant: {cleaned_message[:100]}...")
            if reasoning:
                print(f"    Reasoning: {reasoning[:100]}...")

    return {
        "messages": conversation,
        "metadata": {
            "topic": prompt_or_topic if not is_from_dataset else topic_display,
            "num_turns": num_turns,
            "model_used": args.model,
            "source": "dataset" if is_from_dataset else "topics",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset using MLX models"
    )
    parser.add_argument("--model", required=True, help="MLX model path or identifier")
    parser.add_argument(
        "--output", default="mlx_synthetic_dataset.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--num-convos", type=int, default=50, help="Number of conversations to generate"
    )
    parser.add_argument(
        "--max-turns", type=int, default=6, help="Maximum turns per conversation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum tokens per generation"
    )
    parser.add_argument("--assistant-name", default="Josie", help="Assistant name")
    parser.add_argument(
        "--assistant-role",
        default="an elite MLX assistant created by Gökdeniz Gülmez",
        help="Assistant role description",
    )
    parser.add_argument(
        "--user-role",
        default="a curious MLX developer asking for help",
        help="User role description",
    )
    parser.add_argument(
        "--system-prompt", default="", help="Custom system prompt for assistant"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print conversations instead of saving"
    )
    parser.add_argument(
        "--file-mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="Whether to overwrite existing file or append to it (default: overwrite)",
    )

    # Data source options
    parser.add_argument(
        "--topics",
        nargs="+",
        help="List of topics to generate conversations for (creates prompts from scratch)",
    )
    parser.add_argument(
        "--hf-dataset",
        help='Hugging Face dataset name (e.g., "microsoft/orca-math-word-problems-200k")',
    )
    parser.add_argument("--jsonl-file", help="Path to local JSONL file with prompts")
    parser.add_argument(
        "--combine-sources",
        action="store_true",
        help="Combine multiple data sources (topics + dataset)",
    )
    # Dataset options
    parser.add_argument(
        "--hf-split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to load from dataset"
    )

    args = parser.parse_args()

    print(f"🚀 Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    # Load prompts/topics based on source(s)
    all_prompts = []
    all_sources = []

    # Load from topics (scratch generation)
    if args.topics:
        all_prompts.extend(args.topics)
        all_sources.extend(["topic"] * len(args.topics))
        print(f"📝 Using {len(args.topics)} provided topics for scratch generation")

    # Load from Hugging Face dataset
    if args.hf_dataset:
        hf_prompts = load_prompts_from_hf(
            args.hf_dataset, args.hf_split, args.max_samples
        )
        if hf_prompts:
            all_prompts.extend(hf_prompts)
            all_sources.extend(["hf_dataset"] * len(hf_prompts))
        elif not args.topics and not args.jsonl_file:
            print("❌ Failed to load prompts from Hugging Face dataset. Exiting.")
            sys.exit(1)

    # Load from local JSONL file
    if args.jsonl_file:
        jsonl_prompts = load_prompts_from_jsonl(args.jsonl_file, args.max_samples)
        if jsonl_prompts:
            all_prompts.extend(jsonl_prompts)
            all_sources.extend(["jsonl_file"] * len(jsonl_prompts))
        elif not args.topics and not args.hf_dataset:
            print("❌ Failed to load prompts from JSONL file. Exiting.")
            sys.exit(1)

    # Validate that we have at least one source
    if not all_prompts:
        print(
            "❌ No prompts or topics available. Please provide --topics, --hf-dataset, or --jsonl-file"
        )
        sys.exit(1)

    # Check for multiple sources without combine flag
    unique_sources = set(all_sources)
    if len(unique_sources) > 1 and not args.combine_sources:
        print(
            "⚠️  Multiple data sources detected. Use --combine-sources to mix them, or specify only one source."
        )
        print(f"   Sources found: {', '.join(unique_sources)}")
        sys.exit(1)

    print(
        f"📊 Generating {args.num_convos} conversations from {len(all_prompts)} available prompts/topics"
    )
    if args.combine_sources:
        print(f"🔀 Combining sources: {', '.join(unique_sources)}")

    dataset = []
    for i in range(args.num_convos):
        prompt_or_topic = all_prompts[i % len(all_prompts)]
        source_type = all_sources[i % len(all_sources)]
        is_from_dataset = source_type != "topic"

        print(
            f"🧠 Generating conversation {i+1}/{args.num_convos} (source: {source_type})"
        )

        try:
            conversation = generate_conversation(
                model, tokenizer, args, prompt_or_topic, is_from_dataset
            )
            dataset.append(conversation)

            if args.dry_run:
                print("=" * 80)
                print(json.dumps(conversation, indent=2))
                print("=" * 80)

        except Exception as e:
            print(f"❌ Error generating conversation: {e}")
            continue

    if not args.dry_run and dataset:
        print(f"💾 Saving {len(dataset)} conversations to {args.output}")

        # Determine file mode
        file_mode = "a" if args.file_mode == "append" else "w"

        with open(args.output, file_mode) as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")

        action = "appended to" if args.file_mode == "append" else "saved to"
        print(
            f"✅ Dataset generation complete! {len(dataset)} conversations {action} {args.output}"
        )
    elif args.dry_run:
        print(f"🔍 Dry run complete. Generated {len(dataset)} conversations.")
    else:
        print("❌ No conversations were generated successfully.")


if __name__ == "__main__":
    main()
