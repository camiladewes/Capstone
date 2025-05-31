def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    import lightgbm as lgb
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)
    if params is None:
        params = {'objective': 'regression', 'learning_rate': 0.05, 'metric': 'l1'}
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    return model
