from datasets import concatenate_datasets, load_dataset

def remove_proofs(example):
    return example['problem_type'] != 'converted_proof'

dataset = load_dataset("nvidia/OpenMathReasoning")

dataset['cot'] = dataset['cot'].remove_columns(
    ['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle']
)
dataset['additional_problems'] = dataset['additional_problems'].remove_columns(
    ['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle']
)
full_data = concatenate_datasets([dataset['cot'], dataset['additional_problems']])
full_data = full_data.filter(remove_proofs, num_proc=20)

full_data.to_json("math-problems.jsonl")