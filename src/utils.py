from scipy.stats import kendalltau

def kendalls_tau(prefs, preds):
    # Using scipy's built-in Kendall's Tau implementation
    tau, _ = kendalltau(prefs, preds)
    return tau