def get_y_pred_true(model, test_gen, return_numpy=True):
    """
    Computes y_pred and returns its corresponding t_true as well
    Args:
        model:
        test_gen:
        return_numpy:

    Returns: y_pred and y_true

    """
    import numpy as np
    print('[INFO] Making  prediction ')
    y_pred, y_true = [], []
    for x, y in test_gen:
        predictions = model.predict(x, batch_size=1)
        temp = np.argmax(predictions, axis=1)
        y = np.argmax(y, axis=1)
        y_pred.extend(temp)
        y_true.extend(y)
    if return_numpy:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
    return y_pred, y_true
