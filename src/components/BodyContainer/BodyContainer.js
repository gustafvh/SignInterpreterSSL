import React, { Component } from "react";
import firebase from "../../firebase";
import "./LoadingIcon.scss";
import "./UploadImage.scss";
import oneSign from "../../assets/icons/one-finger.png";
import twoSign from "../../assets/icons/two-fingers.png";
import threeSign from "../../assets/icons/three-fingers.png";

// import image from "../../assets/images/image.png";
import ImageWithSettings from "./ImageWithSettings/ImageWithSettings.js";
import uploadIcon from "../../assets/icons/upload-icon.png";
import axios from "axios";

export default class UploadImage extends Component {
  constructor(props) {
    super(props);

    this.state = {
      uploadedFileUrl: "",
      loading: false,
      imageHasUploaded: false,
      imageName: "",
      googleResponse: undefined,
      similairImagesUrls: [],
      imageLabels: [],
      responseFromAPI: '',
      imageFile: null
    };
  }

  getUploadedFileAsBinary = event => {
    this.setState({
      imageFile: event.target.files[0],
    });
    console.log(event.target.files[0]);
  };

  sendToAPI = () => {
    try {
      this.setState({
        loading: true
      });
      const data = new FormData()
      data.append('image', this.state.imageFile)
      axios.post("http://localhost:5000/predict", data, {}).then(response => {
        this.setState({
          responseFromAPI: response,
          loading: false
        });
        console.log(response.data.predictions)
      })
      this.props.changeCurrentStep(2);
    }
    catch(error) {
      console.log(error)
    }

  }

  uploadToFirebaseStorage = async event => {
    this.setState({
      loading: true
    });

    let fileName = event.target.value;
    fileName = fileName.slice(12);

    let fileToUpload = event.target.files[0];
    let refNameInFirebase = firebase
      .storage()
      .ref("UploadedImages/ppicture.jpg");

    const snapshot = await refNameInFirebase.put(fileToUpload);
    const imageUrl = await snapshot.ref.getDownloadURL();

    this.setState({
      uploadedFileUrl: imageUrl,
      loading: false,
      gettingAPIResponse: false,
      imageHasUploaded: true,
      imageName: fileName
    });
  };

  submitToCloudVisionAPI = async () => {
    try {
      this.setState({ gettingAPIResponse: true });
      let imageUrl = this.state.uploadedFileUrl;
      let body = JSON.stringify({
        requests: [
          {
            features: [
              { type: "LABEL_DETECTION", maxResults: 5 },
              { type: "FACE_DETECTION", maxResults: 5 },
              //   { type: "IMAGE_PROPERTIES", maxResults: 5 },
              { type: "WEB_DETECTION", maxResults: 30 }
            ],
            image: {
              source: {
                imageUri: imageUrl
              }
            }
          }
        ]
      });
      let response = await fetch(
        "https://vision.googleapis.com/v1/images:annotate?key=" +
        process.env.GOOGLE_CLOUD_VISION_API_KEY,
        {
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json"
          },
          method: "POST",
          body: body
        }
      );
      let responseJson = await response.json();

      let similairImagesUrls = [];
      let imageLabels = [];

      for (let i = 0; i < 9; i++) {
        if (
          responseJson.responses[0].webDetection.visuallySimilarImages[i] !==
          undefined
        ) {
          similairImagesUrls.push(
            responseJson.responses[0].webDetection.visuallySimilarImages[i].url
          );
          imageLabels.push(
            responseJson.responses[0].labelAnnotations[0].description
          );
        }
      }

      this.setState({
        googleResponse: responseJson,
        gettingAPIResponse: false,
        similairImagesUrls: similairImagesUrls,
        imageLabels: imageLabels
      });

      this.props.changeCurrentStep(2);
    } catch (error) {
      console.log(error);
    }
  };

  render() {
    return (
      <div>
        {this.state.loading ? (
          <div className="loading-icon__container">
            <div className="lds-ellipsis">
              <div></div>
              <div></div>
              <div></div>
              <div></div>
            </div>
          </div>
        ) : this.state.imageHasUploaded ? (
          <div>
            <ImageWithSettings
              uploadedFileUrl={this.state.uploadedFileUrl}
              imageName={this.state.imageName}
              gettingAPIResponse={this.state.gettingAPIResponse}
              submitToCloudVisionAPI={this.submitToCloudVisionAPI}
              googleResponse={this.state.googleResponse}
              similairImagesUrls={this.state.similairImagesUrls}
              imageLabels={this.state.imageLabels}
            />
          </div>
        ) : (
          <div style={{justifyContent: "center"}}>
            <img className="sign-one" src={oneSign} alt="sign-one" style={{height: "80px", marginLeft: "5px"}}/>
            <img className="sign-two" src={twoSign} alt="sign-two" style={{height: "80px"}}/>
            <img className="sign-three" src={threeSign} alt="sign-three" style={{height: "80px"}}/>
            {/* "Icons made by Freepik from www.flaticon.com" */}
            <button type="button" onClick={this.sendToAPI}>Translate!</button>
            <label htmlFor="file-upload" className="upload-file__button">    {/* Is what is visible*/}
              Select Image<img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={uploadIcon}/>
            </label>
            <input id="file-upload" className="upload-file__button"
                   type="file" accept="image/*"
                   onChange={this.getUploadedFileAsBinary}/>
            {/* Is hidden via scss-file but contains the logic*/}
          </div>
        )}
      </div>
    );
  }
}
