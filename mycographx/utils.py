"""Small compatibility helpers required by the counterfactual rulebook."""


def one_hot(value, categories):
    index = categories.index(value)
    return [int(position == index) for position in range(len(categories))]
