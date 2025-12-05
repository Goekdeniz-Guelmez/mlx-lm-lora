#!/usr/bin/env python3
"""
Synthetic Prompt Generator for MLX-LM-LoRA
  
Generate high-quality synthetic prompt datasets using MLX-LM batch generation.
Supports topic-based generation with optional document grounding.
"""
  
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Dict
import pyarrow as pa
import pyarrow.parquet as pq
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm
  
  
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that generates diverse, high-quality human prompts for training language models.
  
Your task is to create realistic user prompts that someone might ask about the given topic. The prompts should:
- Be natural and varied in style (questions, requests, tasks)
- Range from simple to complex
- Cover different aspects of the topic
- Be suitable for instruction-following training
- Be self-contained and clear
- When document context is provided, incorporate relevant details without straying from the main topic
  
You must respond with a valid JSON object in this exact format:
{"user_prompt": "the generated prompt here"}
  
Only output valid JSON, nothing else, no additional texts or characters start directly with the json object."""
  
  
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic prompt datasets using MLX-LM-LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter)
  
    # Core arguments
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        help="List of topics to generate prompts for (e.g., 'ML' 'politics' 'web security')"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Directory containing PDF, TXT, and MD files for grounding (optional)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Josiefied-Qwen3-4B-Instruct-2507-abliterated-v1-8bit",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (uses default if not provided)"
    )
  
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--valid-split",
        type=float,
        default=None,
        help="Validation split ratio (e.g., 0.1 for 10%%, None to disable)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="Test split ratio (e.g., 0.1 for 10%%, None to disable)"
    )
  
    # Generation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling parameter"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=1,
        help="Minimum tokens to keep"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=0.0,
        help="XTC probability"
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="XTC threshold"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
  
    return parser.parse_args()
  
  
def load_documents(docs_dir: str) -> List[Dict[str, str]]:
    """Load all supported documents from directory."""
    documents = []
    docs_path = Path(docs_dir)
  
    if not docs_path.exists():
        print(f"Warning: Document directory '{docs_dir}' does not exist")
        return documents
  
    # Supported file extensions
    for ext in ["*.txt", "*.md", "*.pdf"]:
        for file_path in docs_path.rglob(ext):
            try:
                if ext == "*.pdf":
                    # Requires PyMuPDF or similar
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(file_path)
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        doc.close()
                    except ImportError:
                        print(f"Warning: PyMuPDF not installed, skipping {file_path}")
                        continue
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
  
                if text.strip():
                    documents.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "content": text
                    })
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
  
    return documents
  
  
def create_generation_prompt(
    topic: Optional[str] = None,
    section: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """Create the prompt for generating synthetic prompts."""
    
    # Case 1: Document-based prompt
    if section:
        # Truncate if too long
        max_context = 2000
        if len(section) > max_context:
            section = section[:max_context] + "..."
        
        user_message = f"""
Context from document:
{section}

Based on this context, generate a diverse, realistic user prompt that someone might ask. The prompt should reference concepts from the context.

Respond with valid JSON only:
{{"user_prompt": "your generated prompt here"}}"""
    
    # Case 2: Topic-based prompt
    elif topic:
        user_message = f"""Topic: {topic}

Generate a diverse, realistic user prompt that someone might ask about this topic. The prompt should be natural and varied in style.

Respond with valid JSON only:
{{"user_prompt": "your generated prompt here"}}"""
    
    else:
        raise ValueError("Either topic or section must be provided")
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


def clean_latex_for_json(text: str) -> str:
    import re
    
    # First try basic cleanup - double all backslashes
    # This handles cases like \theta -> \\theta, \mathbb -> \\mathbb, etc.
    escaped_text = text.replace('\\', '\\\\')
    
    # Fix any cases where we accidentally quadrupled backslashes that were already escaped
    escaped_text = escaped_text.replace('\\\\\\\\', '\\\\')
    
    return escaped_text
  
  
def generate_dataset(args):
    if not args.topics and not args.docs_dir:
        raise ValueError("Either --topics or --docs-dir must be specified")
  
    random.seed(args.seed)
  
    os.makedirs(args.output_dir, exist_ok=True)
  
    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load(path_or_hf_repo=args.model)
  
    # Load documents if provided
    documents = []
    if args.docs_dir:
        print(f"Loading documents from: {args.docs_dir}")
        documents = load_documents(args.docs_dir)
        print(f"Loaded {len(documents)} documents")
        if not documents:
            print("Warning: No documents loaded, falling back to topics-only generation")
  
    # Use custom or default system prompt
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
  
    # Generate samples
    all_samples = []
    num_generated = 0
  
    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        min_tokens_to_keep=args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )
  
    # Calculate batches needed
    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
  
    print(f"Generating {args.num_samples} samples in approximately {total_batches} batches...")
    if args.docs_dir and documents:
        print(f"Using document-based generation")
    if args.topics:
        print(f"Using topics: {', '.join(args.topics)}")
  
    with tqdm(total=args.num_samples, desc="Generating prompts") as pbar:
        while num_generated < args.num_samples:
            batch_prompts = []
            batch_metadata = []
  
            # Create batch
            for _ in range(min(args.batch_size, args.num_samples - num_generated)):
                # Decide whether to use documents or topics
                use_documents = args.docs_dir and documents and (not args.topics or random.random() < 0.5)
                
                if use_documents:
                    # Document-based generation - set topic to None
                    topic = None
                    doc = random.choice(documents)
                    # Extract random section
                    lines = doc["content"].split("\n")
                    if len(lines) > 10:
                        start = random.randint(0, max(0, len(lines) - 10))
                        section = "\n".join(lines[start:start+10])
                    else:
                        section = doc["content"]
                else:
                    # Topic-based generation - set section to None
                    if not args.topics:
                        raise ValueError("No topics provided and no documents available")
                    topic = random.choice(args.topics)
                    section = None
                
                # Create generation prompt
                messages = create_generation_prompt(topic, section, system_prompt)
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
                
                batch_prompts.append(prompt_tokens)
                batch_metadata.append({
                    "topic": topic,
                    "section": section
                })
  
            # Generate batch
            result = batch_generate(
                model,
                tokenizer,
                batch_prompts,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False
            )
  
            # Process results
            for text, metadata in zip(result.texts, batch_metadata):
                # Parse JSON response
                try:
                    # Clean up response (remove markdown code blocks if present)
                    cleaned_text = text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()

                    cleaned_text = cleaned_text.replace('\\\\', '\\\\\\\\')
                    cleaned_text = cleaned_text.replace('\\', '\\\\')
                    cleaned_text = cleaned_text.replace('\\\\\\\\', '\\\\')
                    
                    # Parse JSON
                    parsed = json.loads(cleaned_text)
                    user_prompt = parsed.get("user_prompt", "")
                    
                    if not user_prompt:
                        print(f"Warning: Empty user_prompt in response, skipping")
                        continue
                    
                    sample = {
                        "prompt": user_prompt,
                        "section": metadata["section"],
                        "topic": metadata["topic"]
                    }
                    all_samples.append(sample)
                    num_generated += 1
                    pbar.update(1)
                    
                    if num_generated >= args.num_samples:
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON response: {e}")
                    print(f"Response was: {text[:200]}...")
                    continue
  
    # Shuffle samples
    random.shuffle(all_samples)
  
    # Split dataset
    train_samples = all_samples
    valid_samples = []
    test_samples = []
  
    if args.test_split:
        test_size = int(len(all_samples) * args.test_split)
        test_samples = all_samples[:test_size]
        train_samples = all_samples[test_size:]
  
    if args.valid_split:
        valid_size = int(len(train_samples) * args.valid_split)
        valid_samples = train_samples[:valid_size]
        train_samples = train_samples[valid_size:]
  
    # Save datasets
    def save_split(samples: List[Dict], split_name: str):
        if not samples:
            return
  
        # Save JSONL
        jsonl_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(jsonl_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
  
        # Save Parquet
        parquet_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        table = pa.Table.from_pylist(samples)
        pq.write_table(table, parquet_path)
  
        print(f"Saved {len(samples)} samples to {split_name}.{{jsonl,parquet}}")
  
    save_split(train_samples, "train")
    save_split(valid_samples, "valid")
    save_split(test_samples, "test")
  
    # Generate summary statistics
    topic_count = {}
    doc_count = 0
    
    for sample in all_samples:
        if sample["topic"]:
            topic = sample["topic"]
            topic_count[topic] = topic_count.get(topic, 0) + 1
        else:
            doc_count += 1
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(all_samples)}")
    print(f"  - Document-based: {doc_count}")
    if topic_count:
        print(f"  - Topic-based: {sum(topic_count.values())}")
        for topic, count in topic_count.items():
            print(f"    - {topic}: {count}")
    print(f"Splits: Train: {len(train_samples)}, Valid: {len(valid_samples)}, Test: {len(test_samples)}")
  
  
def main():
    args = parse_args()
    generate_dataset(args)
  
  
if __name__ == "__main__":
    main()