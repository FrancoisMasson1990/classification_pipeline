## AI Dev Interview Test / Classification Pipeline

Build a Classification Pipeline that cleans the [provided dataset](data/iris.csv) and trains a model. The resulting model should then be served via a RESTful API.

Requirments:
- Your pipeline should contain at least one custom preprocessing step implemented with numpy or pandas. All other steps can be implemented with the ML library of your choice.


Functional requirements:
- Build a classification pipeline.
- Train a model using your pipeline and the provided IRIS dataset.
- Serve the model via a `/predictions` endpoint.
	```
  curl -X POST "localhost:5000/predictions" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"feat_1\":val_1,\"feat_2\":val_2}"
  ```

Non-functional requirements:
- must use python
- must not use any REST helper library. Ex: Flask-RESTful, Connexion
- must use Git for version control

How to Submit:
- Send a zip file containing all source code to your HR representative.