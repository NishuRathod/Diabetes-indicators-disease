import joblib
model = joblib.load("E:/Diabetes-indicators-disease/result/model.pkl")
print(type(model))
print(hasattr(model, "predict_proba"))
