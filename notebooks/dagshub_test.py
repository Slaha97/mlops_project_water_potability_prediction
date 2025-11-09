import dagshub
dagshub.init(repo_owner='Slaha97', repo_name='mlops_project_water_potability_prediction', mlflow=True)

import mlflow

mlflow.set_tracking_uri("https://dagshub.com/Slaha97/mlops_project_water_potability_prediction.mlflow")

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)