# email-spam-classifier
Email spam classifier, implemented as REST service. Uses pre-trained TF-IDF model.

How to use it:
If not installed, installe a pipenv by 'pip install pipenv'. 
Unpack the learning set in the 'training_data' folder.
Assuming you're in the project's root folder, run 'pipenv install' followed by 'ipenv shell' to prepare a running environment and set up dependencies. 
Run 'python app/create_model.py'. It will run few minutes and will create and traain the TF-IDF model in the 'app' folder.
Run './bootstrap.sh'
You're ready to try the service
Note that 'post_data.json' contains a sample JSON data to try with the service, the service receives an array of strings (email message bodies) and returns an array of 1 and 0 for spam/not spam.
A sample command to try the service:
curl -i -X POST 127.0.0.1:5000/classify \
  -H "Content-Type: application/json" \
  --data-binary "@post_data.json"
