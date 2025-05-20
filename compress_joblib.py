import joblib
meta = joblib.load("metadata.joblib")  
joblib.dump(meta, "metadata.joblib", compress=9)
