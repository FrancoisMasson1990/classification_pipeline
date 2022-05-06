## Classification Pipeline

Build a Classification Pipeline that cleans the provided dataset and trains a model. The resulting model should then be served via a RESTful API.

Requirments:
- Your pipeline should contain at least one custom preprocessing step implemented with numpy or pandas. All other steps can be implemented with the ML library of your choice.


Functional requirements:
- Build a classification pipeline.
- Train a model using your pipeline and the provided IRIS dataset.
- Serve the model via a `/predictions` endpoint.
	```
  curl -X POST "localhost:5000/predictions" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"feat_1\":val_1,\"feat_2\":val_2}"


Non-functional requirements:
- must use python
- must not use any REST helper library. Ex: Flask-RESTful, Connexion
- must use Git for version control


### Example of usage:
In a terminal, run the following script in order to train and save the best model
```
python3 ~/classification_pipeline/classification.py
```
Once done, run the following script
```
python3 ~/app.py
```

In another terminal, run
```
curl -X POST -H "Content-type: application/json" -H "accept: application/json" -d "{\"sepal_width\" : \"0.2\", \"petal_length\" : \"0.77\", \"petal_width\" : \"0.54\"}" "localhost:5000/predictions"
  
