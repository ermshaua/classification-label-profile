from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

_SCORE_MAPPING = {
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred)
}


def map_scores(score_name):
    """
    Map score names to score functions.

    Parameters
    ----------
    score_name : str
        Name of the score to be mapped. Valid score names are: "f1", "precision", "recall", "accuracy".

    Returns
    -------
    score_func : callable
        The score function corresponding to the input score name.

    Raises
    ------
    ValueError
        If the input score name is not a valid score name.

    Notes
    -----
    This function maps the input score name to its corresponding score function from the score
    function dictionary `_SCORE_MAPPING`, which maps score names to score functions. If the input score
    name is not a valid score name, a `ValueError` is raised.

    Examples
    --------
    >>> score_func = map_scores('f1')
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> score = score_func(y_true, y_pred)
    """
    if score_name not in _SCORE_MAPPING:
        raise ValueError(
            f"{score_name} is not a valid score. Implementations include: {', '.join(_SCORE_MAPPING.keys())}")

    return _SCORE_MAPPING[score_name]
