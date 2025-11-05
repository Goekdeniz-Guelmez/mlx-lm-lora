import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import gradio as gr
from datasets import load_dataset
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


class PreferenceDatasetCreator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.history = []
        self.current_prompts = []
        self.current_index = 0
        self.auto_next = False

    def load_model(self, repo_id: str, progress=gr.Progress()) -> str:
        """Load MLX model from HuggingFace repository"""
        try:
            if not repo_id.strip():
                return "❌ Please enter a valid repository ID"

            progress(0.1, desc="Loading model...")
            self.model, self.tokenizer = load(repo_id)
            self.model_name = repo_id
            progress(1.0, desc="Model loaded successfully!")
            return f"✅ Model loaded: {repo_id}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def load_prompts_from_file(self, file_path: str) -> Tuple[str, List[str]]:
        """Load prompts from local JSONL file"""
        try:
            if not file_path or not os.path.exists(file_path):
                return "❌ File not found", []

            prompts = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Try different common prompt fields
                    prompt = data.get(
                        "prompt", data.get("instruction", data.get("text", ""))
                    )
                    if prompt:
                        prompts.append(prompt)

            self.current_prompts = prompts
            self.current_index = 0
            return f"✅ Loaded {len(prompts)} prompts from file", prompts
        except Exception as e:
            return f"❌ Error loading file: {str(e)}", []

    def load_prompts_from_hf(
        self, dataset_id: str, split: str = "train", prompt_field: str = "prompt"
    ) -> Tuple[str, List[str]]:
        """Load prompts from HuggingFace dataset"""
        try:
            if not dataset_id.strip():
                return "❌ Please enter a valid dataset ID", []

            dataset = load_dataset(dataset_id, split=split)
            prompts = [item[prompt_field] for item in dataset if prompt_field in item]

            self.current_prompts = prompts
            self.current_index = 0
            return f"✅ Loaded {len(prompts)} prompts from {dataset_id}", prompts
        except Exception as e:
            return f"❌ Error loading dataset: {str(e)}", []

    def generate_responses(
        self,
        prompt: str,
        system_prompt: str = "",
        system_for_both: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.0,
        top_k: int = 50,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.1,
        progress=gr.Progress(),
    ) -> Tuple[str, str]:
        """Generate two responses using MLX with batch generation and proper chat templates"""
        if not self.model or not self.tokenizer:
            return "❌ No model loaded", "❌ No model loaded"

        try:
            progress(0.1, desc="Preparing prompts...")

            # Prepare conversation templates
            def create_conversation(include_system: bool = True):
                conversation = []

                # Add system message if provided and requested
                if system_prompt.strip() and include_system:
                    conversation.append(
                        {"role": "system", "content": system_prompt.strip()}
                    )

                # Add user message
                conversation.append({"role": "user", "content": prompt.strip()})

                return conversation

            # Create conversations for both responses
            if system_for_both:
                conversation1 = create_conversation(include_system=True)
                conversation2 = create_conversation(include_system=True)
            else:
                conversation1 = create_conversation(
                    include_system=True
                )  # With system prompt
                conversation2 = create_conversation(
                    include_system=False
                )  # Without system prompt

            progress(0.2, desc="Applying chat templates...")

            # Apply chat templates
            try:
                # Check if tokenizer has apply_chat_template method
                if hasattr(self.tokenizer, "apply_chat_template"):
                    # Determine if this is a Qwen model for enable_thinking parameter
                    is_qwen_model = (
                        True
                        if self.model.args.model_type == "qwen3"
                        or self.model.args.model_type == "qwen3_moe"
                        else False
                    )

                    prompt1 = self.tokenizer.apply_chat_template(
                        conversation=conversation1,
                        add_generation_prompt=True,  # Changed to True to add assistant prompt
                        enable_thinking=(
                            False if is_qwen_model else None
                        ),  # Only for Qwen models
                        tokenize=False,
                    )

                    prompt2 = self.tokenizer.apply_chat_template(
                        conversation=conversation2,
                        add_generation_prompt=True,  # Changed to True to add assistant prompt
                        enable_thinking=(
                            False if is_qwen_model else None
                        ),  # Only for Qwen models
                        tokenize=False,
                    )

                    # Remove enable_thinking parameter for non-Qwen models
                    if not is_qwen_model:
                        prompt1 = self.tokenizer.apply_chat_template(
                            conversation=conversation1,
                            add_generation_prompt=True,
                            tokenize=False,
                        )

                        prompt2 = self.tokenizer.apply_chat_template(
                            conversation=conversation2,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                else:
                    # Fallback for tokenizers without chat template support
                    def format_conversation(conv):
                        formatted = ""
                        for message in conv:
                            if message["role"] == "system":
                                formatted += f"System: {message['content']}\n\n"
                            elif message["role"] == "user":
                                formatted += f"User: {message['content']}\n\n"
                        formatted += "Assistant:"
                        return formatted

                    prompt1 = format_conversation(conversation1)
                    prompt2 = format_conversation(conversation2)

            except Exception as template_error:
                # Fallback formatting if chat template fails
                progress(
                    0.25, desc="Chat template failed, using fallback formatting..."
                )

                def format_conversation_fallback(conv):
                    formatted = ""
                    for message in conv:
                        if message["role"] == "system":
                            formatted += (
                                f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                            )
                        elif message["role"] == "user":
                            formatted += (
                                f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                            )
                    formatted += "<|im_start|>assistant\n"
                    return formatted

                prompt1 = format_conversation_fallback(conversation1)
                prompt2 = format_conversation_fallback(conversation2)

            progress(0.3, desc="Generating responses...")

            # Batch generation with MLX
            prompts_batch = [prompt1, prompt2]
            responses = []

            # Create sampler with all parameters
            sampler_kwargs = {
                "temp": temperature,
            }

            # Only add parameters if they have non-default values
            if top_p != 1.0:
                sampler_kwargs["top_p"] = top_p
            if min_p > 0.0:
                sampler_kwargs["min_p"] = min_p
            if top_k > 0:
                sampler_kwargs["top_k"] = top_k
            if xtc_probability > 0.0:
                sampler_kwargs["xtc_probability"] = xtc_probability
                sampler_kwargs["xtc_threshold"] = xtc_threshold
                # Add special tokens for XTC if available
                if hasattr(self.tokenizer, "encode") and hasattr(
                    self.tokenizer, "eos_token_ids"
                ):
                    try:
                        newline_tokens = self.tokenizer.encode("\n")
                        eos_tokens = list(self.tokenizer.eos_token_ids)
                        sampler_kwargs["xtc_special_tokens"] = (
                            newline_tokens + eos_tokens
                        )
                    except:
                        pass  # Skip if tokenizer doesn't support these methods

            sampler = make_sampler(**sampler_kwargs)

            for i, p in enumerate(prompts_batch):
                progress(0.3 + (i * 0.35), desc=f"Generating response {i+1}/2...")

                try:
                    response = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=p,
                        sampler=sampler,
                        max_tokens=max_tokens,
                        verbose=False,
                    )

                    # Clean up the response (remove the original prompt if it's included)
                    if isinstance(response, str):
                        # If the response includes the prompt, try to extract just the generated part
                        if p in response:
                            response = response.replace(p, "").strip()

                        # Remove common chat template artifacts
                        response = response.replace("<|im_end|>", "").strip()
                        response = response.replace("<|endoftext|>", "").strip()

                    responses.append(response)

                except Exception as gen_error:
                    error_response = (
                        f"❌ Generation error for response {i+1}: {str(gen_error)}"
                    )
                    responses.append(error_response)

            progress(1.0, desc="Generation complete!")
            return responses[0], responses[1]

        except Exception as e:
            error_msg = f"❌ Generation error: {str(e)}"
            return error_msg, error_msg

    def save_preference(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        preference: str,
        system_prompt: str = "",
        system_for_both: bool = True,
    ) -> str:
        """Save the preference choice to history"""
        if preference == "No preference":
            return "⚠️ Please select a preference"

        sample = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "system_for_both": system_for_both,
            "response_a": response_a,
            "response_b": response_b,
            "preference": preference,
            "chosen": response_a if preference == "Response A" else response_b,
            "rejected": response_b if preference == "Response A" else response_a,
        }

        self.history.append(sample)
        return f"✅ Preference saved! Total samples: {len(self.history)}"

    def export_dataset(self, file_path: str) -> str:
        """Export the preference dataset to JSONL file"""
        try:
            if not self.history:
                return "❌ No data to export"

            if not file_path.strip():
                file_path = f"preference_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

            # Ensure .jsonl extension
            if not file_path.endswith(".jsonl"):
                file_path += ".jsonl"

            with open(file_path, "w", encoding="utf-8") as f:
                for sample in self.history:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            return f"✅ Dataset exported to {file_path} ({len(self.history)} samples)"
        except Exception as e:
            return f"❌ Export error: {str(e)}"

    def get_current_prompt(self) -> str:
        """Get current prompt from loaded prompts"""
        if not self.current_prompts or self.current_index >= len(self.current_prompts):
            return ""
        return self.current_prompts[self.current_index]

    def next_prompt(self) -> Tuple[str, str]:
        """Move to next prompt"""
        if self.current_prompts and self.current_index < len(self.current_prompts) - 1:
            self.current_index += 1
            return (
                self.get_current_prompt(),
                f"Prompt {self.current_index + 1}/{len(self.current_prompts)}",
            )
        return "", "No more prompts"

    def previous_prompt(self) -> Tuple[str, str]:
        """Move to previous prompt"""
        if self.current_prompts and self.current_index > 0:
            self.current_index -= 1
            return (
                self.get_current_prompt(),
                f"Prompt {self.current_index + 1}/{len(self.current_prompts)}",
            )
        return "", "At first prompt"

    def get_history_display(self) -> str:
        """Format history for display"""
        if not self.history:
            return "No samples yet"

        display = []
        for i, sample in enumerate(self.history[-10:], 1):  # Show last 10
            display.append(f"**Sample {len(self.history)-10+i}:**")
            display.append(f"Prompt: {sample['prompt'][:100]}...")
            display.append(f"Preference: {sample['preference']}")
            display.append("---")

        return "\n".join(display)


# Initialize the creator
creator = PreferenceDatasetCreator()


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="MLX Preference Dataset Creator", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 🎯 MLX Preference Dataset Creator")
        gr.Markdown(
            "Create preference datasets using MLX models with side-by-side response comparison"
        )

        with gr.Tab("🤖 Model Setup"):
            with gr.Row():
                model_repo = gr.Textbox(
                    label="HuggingFace Model Repository ID",
                    placeholder="mlx-community/Josiefied-Qwen3-0.6B-abliterated-v1-4bit",
                    value="mlx-community/Josiefied-Qwen3-0.6B-abliterated-v1-4bit",
                )
                load_model_btn = gr.Button("Load Model", variant="primary")

            model_status = gr.Textbox(label="Model Status", interactive=False)

            load_model_btn.click(
                creator.load_model, inputs=[model_repo], outputs=[model_status]
            )

        with gr.Tab("📝 Prompt Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Manual Prompt Entry")
                    manual_prompt = gr.Textbox(
                        label="Enter Prompt",
                        lines=3,
                        placeholder="Type your prompt here...",
                    )

                with gr.Column():
                    gr.Markdown("### Load from File")
                    file_path = gr.Textbox(
                        label="JSONL File Path", placeholder="/path/to/prompts.jsonl"
                    )
                    load_file_btn = gr.Button("Load File")

                with gr.Column():
                    gr.Markdown("### Load from HuggingFace")
                    with gr.Row():
                        hf_dataset_id = gr.Textbox(
                            label="Dataset ID",
                            placeholder="mlx-community/Human-Like-DPO",
                            value="mlx-community/Human-Like-DPO",
                        )
                        hf_split = gr.Textbox(label="Split", value="train")
                    hf_prompt_field = gr.Textbox(label="Prompt Field", value="prompt")
                    load_hf_btn = gr.Button("Load Dataset")

            prompt_status = gr.Textbox(label="Prompt Loading Status", interactive=False)

            with gr.Row():
                prev_btn = gr.Button("← Previous")
                prompt_counter = gr.Textbox(label="Position", interactive=False)
                next_btn = gr.Button("Next →")

            # Event handlers for prompt loading
            load_file_btn.click(
                creator.load_prompts_from_file,
                inputs=[file_path],
                outputs=[prompt_status, gr.State()],
            )

            load_hf_btn.click(
                creator.load_prompts_from_hf,
                inputs=[hf_dataset_id, hf_split, hf_prompt_field],
                outputs=[prompt_status, gr.State()],
            )

        with gr.Tab("⚡ Generation & Comparison"):
            with gr.Row():
                with gr.Column():
                    current_prompt = gr.Textbox(
                        label="Current Prompt", lines=3, value=""
                    )

                    with gr.Row():
                        system_prompt = gr.Textbox(
                            label="System Prompt (Optional)",
                            lines=2,
                            placeholder="You are a helpful assistant...",
                        )
                        system_for_both = gr.Checkbox(
                            label="Apply system prompt to both responses",
                            value=True,
                            info="If unchecked, system prompt applies only to Response A",
                        )

                    with gr.Accordion("🎛️ Sampling Parameters", open=False):
                        with gr.Row():
                            max_tokens = gr.Slider(
                                label="Max Tokens",
                                minimum=50,
                                maximum=2048,
                                value=512,
                                step=50,
                            )
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                            )

                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                            )
                            min_p = gr.Slider(
                                label="Min-p",
                                minimum=0.0,
                                maximum=0.5,
                                value=0.0,
                                step=0.01,
                            )

                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k", minimum=0, maximum=200, value=50, step=1
                            )

                        with gr.Row():
                            xtc_probability = gr.Slider(
                                label="XTC Probability",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.01,
                                info="Probability of applying XTC sampling",
                            )
                            xtc_threshold = gr.Slider(
                                label="XTC Threshold",
                                minimum=0.01,
                                maximum=1.0,
                                value=0.1,
                                step=0.01,
                                info="Threshold for XTC sampling",
                            )

                    generate_btn = gr.Button(
                        "🚀 Generate Responses", variant="primary", size="lg"
                    )

                with gr.Column():
                    auto_next_toggle = gr.Checkbox(
                        label="Auto Next Question",
                        value=False,
                        info="Automatically move to next question after saving preference",
                    )

                    preference_choice = gr.Radio(
                        label="Which response do you prefer?",
                        choices=["Response A", "Response B", "No preference"],
                        value="No preference",
                    )

                    with gr.Row():
                        save_preference_btn = gr.Button(
                            "💾 Save Preference", variant="primary"
                        )
                        manual_next_btn = gr.Button("➡️ Next Question")

                    preference_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🅰️ Response A")
                    response_a = gr.Textbox(
                        label="Response A", lines=10, interactive=False
                    )

                with gr.Column():
                    gr.Markdown("### 🅱️ Response B")
                    response_b = gr.Textbox(
                        label="Response B", lines=10, interactive=False
                    )

            # Generation event handler
            generate_btn.click(
                creator.generate_responses,
                inputs=[
                    current_prompt,
                    system_prompt,
                    system_for_both,
                    max_tokens,
                    temperature,
                    top_p,
                    min_p,
                    top_k,
                    xtc_probability,
                    xtc_threshold,
                ],
                outputs=[response_a, response_b],
            )

            # Navigation event handlers
            def update_current_prompt():
                prompt = creator.get_current_prompt()
                counter = (
                    f"Prompt {creator.current_index + 1}/{len(creator.current_prompts)}"
                    if creator.current_prompts
                    else "No prompts loaded"
                )
                return prompt, counter

            def handle_next():
                prompt, counter = creator.next_prompt()
                return prompt, counter

            def handle_prev():
                prompt, counter = creator.previous_prompt()
                return prompt, counter

            next_btn.click(handle_next, outputs=[current_prompt, prompt_counter])

            prev_btn.click(handle_prev, outputs=[current_prompt, prompt_counter])

            # Auto-update current prompt when manual prompt is used
            manual_prompt.change(
                lambda x: x, inputs=[manual_prompt], outputs=[current_prompt]
            )

            # Enhanced preference saving with auto-next logic and automatic generation
            def save_and_maybe_next(
                prompt,
                resp_a,
                resp_b,
                pref,
                sys_prompt,
                sys_for_both,
                auto_next,
                max_tok,
                temp,
                top_p_val,
                min_p_val,
                top_k_val,
                xtc_prob,
                xtc_thresh,
            ):
                # Save current preference
                status = creator.save_preference(
                    prompt, resp_a, resp_b, pref, sys_prompt, sys_for_both
                )

                if auto_next and pref != "No preference":
                    # Move to next prompt
                    next_prompt_text, counter = creator.next_prompt()

                    if next_prompt_text:  # If there's a next prompt
                        # Generate responses for the new prompt
                        new_resp_a, new_resp_b = creator.generate_responses(
                            next_prompt_text,
                            sys_prompt,
                            sys_for_both,
                            max_tok,
                            temp,
                            top_p_val,
                            min_p_val,
                            top_k_val,
                            xtc_prob,
                            xtc_thresh,
                        )
                        return (
                            status + " → Auto-generated next responses",
                            next_prompt_text,
                            counter,
                            "No preference",
                            new_resp_a,
                            new_resp_b,
                        )
                    else:
                        return (
                            status + " → No more prompts",
                            next_prompt_text,
                            counter,
                            "No preference",
                            "",
                            "",
                        )
                else:
                    return status, prompt, prompt_counter.value, pref, resp_a, resp_b

            save_preference_btn.click(
                save_and_maybe_next,
                inputs=[
                    current_prompt,
                    response_a,
                    response_b,
                    preference_choice,
                    system_prompt,
                    system_for_both,
                    auto_next_toggle,
                    max_tokens,
                    temperature,
                    top_p,
                    min_p,
                    top_k,
                    xtc_probability,
                    xtc_threshold,
                ],
                outputs=[
                    preference_status,
                    current_prompt,
                    prompt_counter,
                    preference_choice,
                    response_a,
                    response_b,
                ],
            )

            # Manual next with automatic generation
            def manual_next_with_generation(
                sys_prompt,
                sys_for_both,
                max_tok,
                temp,
                top_p_val,
                min_p_val,
                top_k_val,
                xtc_prob,
                xtc_thresh,
            ):
                next_prompt_text, counter = creator.next_prompt()

                if next_prompt_text:  # If there's a next prompt
                    # Generate responses for the new prompt
                    new_resp_a, new_resp_b = creator.generate_responses(
                        next_prompt_text,
                        sys_prompt,
                        sys_for_both,
                        max_tok,
                        temp,
                        top_p_val,
                        min_p_val,
                        top_k_val,
                        xtc_prob,
                        xtc_thresh,
                    )
                    return (
                        next_prompt_text,
                        counter,
                        "No preference",
                        new_resp_a,
                        new_resp_b,
                    )
                else:
                    return next_prompt_text, counter, "No preference", "", ""

            manual_next_btn.click(
                manual_next_with_generation,
                inputs=[
                    system_prompt,
                    system_for_both,
                    max_tokens,
                    temperature,
                    top_p,
                    min_p,
                    top_k,
                    xtc_probability,
                    xtc_threshold,
                ],
                outputs=[
                    current_prompt,
                    prompt_counter,
                    preference_choice,
                    response_a,
                    response_b,
                ],
            )

        with gr.Tab("📊 History & Export"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📜 Recent History")
                    history_display = gr.Textbox(
                        label="Last 10 Samples", lines=15, interactive=False
                    )

                    refresh_history_btn = gr.Button("🔄 Refresh History")

                with gr.Column():
                    gr.Markdown("### 💾 Export Dataset")
                    export_path = gr.Textbox(
                        label="Export File Path",
                        placeholder="preference_dataset.jsonl",
                        value="",
                    )

                    export_btn = gr.Button("📤 Export Dataset", variant="primary")
                    export_status = gr.Textbox(label="Export Status", interactive=False)

                    gr.Markdown("### 📈 Statistics")
                    stats_display = gr.Textbox(
                        label="Dataset Statistics", interactive=False
                    )

            # History and export handlers
            def refresh_history():
                history = creator.get_history_display()
                stats = f"Total samples: {len(creator.history)}\nModel: {creator.model_name}"
                return history, stats

            refresh_history_btn.click(
                refresh_history, outputs=[history_display, stats_display]
            )

            export_btn.click(
                creator.export_dataset, inputs=[export_path], outputs=[export_status]
            )

            # Auto-refresh history when preferences are saved
            save_preference_btn.click(
                refresh_history, outputs=[history_display, stats_display]
            )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DPO dataset using MLX models"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="IP address to set default is localhost e.g. 127.0.0.1",
    )
    parser.add_argument("--server-port", type=int, default=7860, help="Port to runn on")
    parser.add_argument(
        "--share", type=bool, default=True, help="Expose the link to the web"
    )
    parser.add_argument("--debug", type=bool, default=True)
    args = parser.parse_args()

    demo = create_interface()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=True,
    )


if __name__ == "__main__":
    main()
