def train(config, dataset):
    raise NotImplementedError('Implement causal episode splits, action normalization and Smooth L1 training')


def evaluate(config, checkpoint, dataset):
    raise NotImplementedError('Implement offline/closed-loop evaluation with explicit clock and label definitions')
