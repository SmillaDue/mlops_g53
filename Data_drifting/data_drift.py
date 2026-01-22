import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.legacy.report import Report

feature_columns = ["avg_brightness", "contrast", "sharpness", "prediction", "target"]


reference_data = pd.read_csv("data_api/reference_features.csv")[feature_columns]
current_data = pd.read_csv("data_api/inference_features.csv")[feature_columns]

reference_data["prediction"] = reference_data["prediction"].map(
    {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
)
current_data["prediction"] = current_data["prediction"].map(
    {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
)

report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("data_drift.html")

from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import TestNumberOfMissingValues
data_test = TestSuite(tests=[TestNumberOfMissingValues()])
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
print("All tests passed: ", result["summary"]["all_passed"])
