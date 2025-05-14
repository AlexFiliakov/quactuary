"""
(Phase 3 features)
Machine learning module.

Start with GLMs and extend to additional Machine Learning algorithms for predictive modeling.

GLMModel: A wrapper for fitting predictive models (frequency/severity modeling with rating factors, loss cost modeling, etc.).
Rather than writing a new GLM solver, quActuary will utilize statsmodels or scikit-learn under the hood.
For example, it could wrap statsmodels.api.GLM for a Tweedie regression on claim counts or sizes.
The interface would allow actuaries to input a pandas DataFrame of exposure data with features and a target
(losses or claim counts) and specify a distribution (Poisson, Gamma, Tweedie, etc.).
Methods like .fit(X, y) would train the model (delegating to statsmodels/sklearn) and store fitted parameters,
and .predict(X_new) would produce pandas Series of predictions.
By conforming to scikit-learn’s estimator API (fit/predict, get_params()/set_params(), etc.),
we ensure compatibility with sklearn tools and pipelines.

Additionally, embedding quantum here could mean exploring quantum machine learning for GLMs in the future,
but initially the focus is on classical implementation with a familiar API.
"""
