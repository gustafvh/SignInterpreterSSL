import React, { Component } from "react";
import UploadImageButton from "./UploadImageButton.js";
import firebase from "../../config/firebase";
import Environment from "../../config/environment";
import "./LoadingIcon.scss";
import "./UploadImage.scss";

// import image from "../../assets/images/image.png";
import ImageWithSettings from "./ImageWithSettings/ImageWithSettings.js";

export default class UploadImage extends Component {
  constructor(props) {
    super(props);

    this.state = {
      uploadedFileUrl: "",
      uploadingImage: false,
      imageHasUploaded: false,
      imageName: "",
      googleResponse: undefined,
      similairImagesUrls: [],
      imageLabels: []
    };
  }

  uploadToFirebaseStorage = async event => {
    this.setState({
      uploadingImage: true
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
      uploadingImage: false,
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
          Environment["GOOGLE_CLOUD_VISION_API_KEY"],
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
        {this.state.uploadingImage ? (
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
          <div>
            {/*<img className="emoji-image" src={emojis} alt="emojis" />*/}
            IMAGE
            <UploadImageButton
              uploadToFirebaseStorage={this.uploadToFirebaseStorage}
            />
          </div>
        )}
      </div>
    );
  }
}
