"""
Machine learning models for predictive actuarial modeling (Phase 3 features).

This module integrates GLM and other ML algorithms (via statsmodels or scikit-learn) for
frequency and severity predictive modeling. Supports fit/predict API compatible with scikit-learn.

Examples:
    >>> from quactuary.future.machine_learning import GLMModel
    >>> glm = GLMModel(distribution='Poisson')
    >>> glm.fit(X_train, y_train)
    >>> preds = glm.predict(X_test)
"""
