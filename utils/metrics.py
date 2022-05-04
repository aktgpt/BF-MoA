def get_metrics(output, target):
    epsilon = 1e-7
    accuracies = [[] for i in range(output.shape[1])]
    f1s = [[] for i in range(output.shape[1])]
    tps = 0
    tns = 0
    fps = 0
    fns = 0
    for i in range(output.shape[1]):
        tp = (target[:, i] * output[:, i]).sum()
        tn = ((1 - target[:, i]) * (1 - output[:, i])).sum()
        fp = ((1 - target[:, i]) * output[:, i]).sum()
        fn = (target[:, i] * (1 - output[:, i])).sum()

        accuracies[i] = ((tp + tn) / (tp + tn + fp + fn)).item()
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1s[i] = (2 * (precision * recall) / (precision + recall + epsilon)).item()
        tps += tp
        tns += tn
        fps += fp
        fns += fn
    precision = tps / (tps + fps + epsilon)
    recall = tps / (tps + fns + epsilon)
    mean_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    mean_accuracy = (tps + tns) / (tps + tns + fps + fns)
    return accuracies, f1s, mean_accuracy.item(), mean_f1.item()
