import json
import re

def load_baby_vocab():
    with open("multimodal/vocab.json", 'r') as f:
        return list(json.load(f).keys()) # return list of baby vocabularies

def vocab_class_filter(class_names, vocab_set, match_type='full'):
    matched_class_names = set()
    for class_name in class_names:
        split_names = set(re.split(r'\W+', class_name))  # Split using regular expression
        if match_type == 'partial':
            if split_names & vocab_set:
                matched_class_names.add(class_name)
        elif match_type == 'full':
            if split_names <= vocab_set:
                matched_class_names.add(class_name)
    return matched_class_names

# def vocab_attr_filter(attr_names, vocab_set, match_type='full'):
#     matched_attr_names = set()
#     for attr_name in attr_names:
#         split_names = set(re.split(r'\W+', attr_name))  # Split using regular expression
#         if match_type == 'partial':
#             if split_names & vocab_set:
#                 matched_attr_names.add(attr_name)
#         elif match_type == 'full':
#             if split_names <= vocab_set:
#                 matched_attr_names.add(attr_name)
#     return matched_attr_names