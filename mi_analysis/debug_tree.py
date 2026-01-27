"""Debug tree parsing issue"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import ast
import sys
sys.path.insert(0, '../src')

from llego.custom.parsing_to_dict import parse_cart_to_dict
from llego.custom.parsing_to_string import parse_dict_to_string

# Load simple dataset
X, y = load_iris(return_X_y=True)
y = (y == 0).astype(int)  # Binary

# Fit forest
rf = RandomForestClassifier(n_estimators=2, max_depth=3, max_samples=0.5, random_state=42)
rf.fit(X, y)

feature_names = ['f0', 'f1', 'f2', 'f3']
tree_dict, _ = parse_cart_to_dict(rf.estimators_[0], feature_names, 'classification')
print('Tree dict:', tree_dict)

# Check types
def check_types(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            check_types(v, path + "/" + str(k))
    else:
        print(f"{path}: {type(d).__module__}.{type(d).__name__} = {d}")

print("\nTypes in tree:")
check_types(tree_dict)

# Try converting to string and back
tree_str = parse_dict_to_string(tree_dict)
print('\nTree string:', tree_str)

# Unescape
parsed_string = tree_str.replace('{{', '{').replace('}}', '}')
print('\nUnescaped:', parsed_string)

# Try literal_eval
try:
    parsed = ast.literal_eval(parsed_string)
    print('\nParsed OK')
except Exception as e:
    print(f'\nParse failed: {e}')
