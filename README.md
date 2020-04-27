# Sign Interpreter Application for Swedish Sign Language
Deep Neural Network for predicting Swedish Sign Language Signs that utilises CNN and Transfer Learning. Contains code for Model, Data Generator and Frontend Application.

By [Gustaf Halvardsson](https://github.com/gustafvh) & [Johanna Peterson](https://github.com/johannakin) 

## Technologies
- Written in **Javascript** using **React**, **HTML** and **CSS**. 
- **Webpack** handles all dependency and builds the final optimized build.
- Post-requests are done using **Axios**
- Deployed via **Heroku** on the domain https://sign-interpreter-ssl.herokuapp.com/ and uses automatic deployment connected to this master-branch.

## Backend
This app uses a seperated backend API in Python that we have also written. The frontend sends a HTTP POST-request to the backend which hosts our trained model (output from our model program) and returns a prediction.
You can view that API repository here: https://github.com/gustafvh/sign-interpreter-ssl-api

## Architecture

<img src="docs/cloud-architecture.png" alt="cloud-architecture"
	title="cloud-architecture" width="600" />


### Node Packages 
The following node-packages were used
```javascript
    "async": "^3.2.0",
    "axios": "^0.19.2",
    "dotenv": "^8.2.0",
    "firebase": "^7.14.1",
    "node-sass": "^4.13.1",
    "normalize.css": "^8.0.1",
    "react": "^16.13.1",
    "react-dom": "^16.13.1",
    "react-scripts": "3.4.1",
    "react-webcam": "^5.0.1"
```

Try the app here: https://sign-interpreter-ssl.herokuapp.com/

## Plans for future work
- Smartphone responsive
- Support user-uploads for training data to improve model-accuracy
