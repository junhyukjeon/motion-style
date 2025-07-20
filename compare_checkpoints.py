import json
import random
import yaml
from sklearn.model_selection import train_test_split

def find_shared_styles(config_paths, n_styles=10):
    style_sets = []

    for config_path in config_paths:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        with open(config["style_json"]) as f:
            full_label_to_ids = json.load(f)

        all_styles = list(full_label_to_ids.keys())
        train_styles, test_styles = train_test_split(
            all_styles,
            test_size=1 - config["train_split"],
            random_state=config["random_seed"]
        )

        if config.get("num_test_styles") is not None:
            rng = random.Random(config["random_seed"])
            test_styles = rng.sample(test_styles, config["num_test_styles"])

        style_sets.append(set(test_styles))

    # Find intersection of all test style sets
    shared_styles = set.intersection(*style_sets)
    print(f"✅ Found {len(shared_styles)} shared styles across all configs")

    # Randomly sample n styles if too many
    if len(shared_styles) >= n_styles:
        shared_styles = random.sample(list(shared_styles), n_styles)
        print(f"🔢 Using {n_styles} styles: {shared_styles}")
    else:
        raise ValueError("Not enough shared styles across configs")

    return shared_styles


if __name__ == "__main__":
    config_paths = [
        "./configs/0.yaml",
        "./configs/1.yaml",
        "./configs/2.yaml",
        "./configs/3.yaml"
    ]

    shared_styles = find_shared_styles(config_paths)

    import pdb; pdb.set_trace()