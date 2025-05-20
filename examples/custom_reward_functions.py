from typing import List, Optional
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

def extract_content_between_tags(text: str, start_tag: str, end_tag: str) -> str:
    """Extract content between given tags"""
    try:
        content = text.split(start_tag)[-1]
        content = content.split(end_tag)[0]
        return content.strip()
    except:
        return ""

@register_reward_function()
def exact_match_reward(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Reward function that gives a high score for exact matches with reference answers.
    
    Args:
        prompts: List of prompt texts
        completions: List of model completion texts
        answer: List of reference answers
        types: Optional list of response types
        
    Returns:
        List of reward scores (0.0 to 2.0)
    """
    if not completions or not answer:
        return [0.0] * len(prompts)
    
    # Simple exact matching
    return [
        2.0 if c.strip() == a.strip() else 0.0 
        for c, a in zip(completions, answer)
    ]

@register_reward_function()
def custom_tag_match_reward(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Reward function that extracts content from custom tags and compares with reference answers.
    
    Args:
        prompts: List of prompt texts
        completions: List of model completion texts
        answer: List of reference answers
        types: Optional list of response types
        
    Returns:
        List of reward scores (0.0 to 2.0)
    """
    if not completions or not answer:
        return [0.0] * len(prompts)
    
    # Extract content from <result> tags
    extracted_results = [extract_content_between_tags(c, "<result>", "</result>") for c in completions]
    
    # Compare with reference answers
    return [
        2.0 if result and expected and result.strip() == expected.strip() else 0.0 
        for result, expected in zip(extracted_results, answer)
    ]

@register_reward_function()
def format_adherence_reward(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Reward function that gives points based on adherence to expected format.
    
    Args:
        prompts: List of prompt texts
        completions: List of model completion texts
        answer: List of reference answers
        types: Optional list of response types
        
    Returns:
        List of reward scores (0.0 to 1.0)
    """
    if not completions:
        return [0.0] * len(prompts)
    
    scores = []
    for completion in completions:
        if not completion:
            scores.append(0.0)
            continue
        
        score = 0.0
        
        # Check for reasoning section
        reasoning_start = completion.find("<reasoning>")
        reasoning_end = completion.find("</reasoning>")
        
        # Check for result section
        result_start = completion.find("<result>")
        result_end = completion.find("</result>")
        
        # Award points for proper tag structure and ordering
        if reasoning_start != -1:
            score += 0.2  # Has opening reasoning tag
        if reasoning_end != -1 and reasoning_start < reasoning_end:
            score += 0.2  # Has closing reasoning tag in correct order
        if result_start != -1 and (reasoning_end == -1 or reasoning_end < result_start):
            score += 0.2  # Has opening result tag in correct order
        if result_end != -1 and result_start < result_end:
            score += 0.2  # Has closing result tag in correct order
            
        # Check if result content exists
        if result_start != -1 and result_end != -1:
            result_content = completion[result_start + 8 : result_end].strip()
            if result_content:
                score += 0.2  # Result tag contains content
        
        scores.append(score)
    
    return scores

@register_reward_function()
def numeric_answer_reward(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """
    Reward function for math problems that extracts and compares numeric answers.
    
    Args:
        prompts: List of prompt texts
        completions: List of model completion texts
        answer: List of reference answers
        types: Optional list of response types
        
    Returns:
        List of reward scores (0.0 to 1.5)
    """
    if not completions or not answer:
        return [0.0] * len(prompts)
    
    # Try to extract numbers from completions
    def extract_number(text):
        # First look for numbers in result tags
        result = extract_content_between_tags(text, "<result>", "</result>")
        if result and result.strip().isdigit():
            return result.strip()
        
        # Then try extracting the last number in the text
        import re
        numbers = re.findall(r'\b\d+\b', text)
        return numbers[-1] if numbers else None
    
    extracted_numbers = [extract_number(c) for c in completions]
    reference_numbers = [a.strip() if a else None for a in answer]
    
    # Compare with reference answers
    return [
        1.5 if num and ref and num == ref else 0.0 
        for num, ref in zip(extracted_numbers, reference_numbers)
    ] 