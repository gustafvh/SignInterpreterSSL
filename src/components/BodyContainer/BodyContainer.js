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

export default class BodyContainer extends Component {
  constructor(props) {
    super(props);

    this.state = {
      loading: false,
      fileUploading: false,
      responseFromAPI: null,
      predictions: null,
      imageFile: null,
      imageFileObjectURL: null,

    };
  }

  getUploadedFileAsBinary = event => {
    this.setState({
      imageFile: event.target.files[0],
      imageFileObjectURL: URL.createObjectURL(event.target.files[0])
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
      axios.post("http://35.198.151.110/predict", data, {}).then(response => {
        this.setState({
          responseFromAPI: response,
          predictions: response.data.predictions,
          loading: false
        });
        console.log(response.data.predictions)
      }).then(this.setState({ loading: false }));
      this.props.changeCurrentStep(2);
    }
    catch(error) {
      console.log(error)
    }

  }

  renderloadingElement = () => {
    return (
              <div className="loading-icon__container">
                <div className="lds-ellipsis">
                  <div></div>
                  <div></div>
                  <div></div>
                  <div></div>
                </div>
              </div>
    )
  }

  render() {
    return (
      <div>
        {this.state.loading ? this.renderloadingElement()
         : this.state.responseFromAPI ? (
          <div>
            <ImageWithSettings
              imageFileObjectURL={this.state.imageFileObjectURL}
              imageFile={this.state.imageFile}
              predictions={this.state.predictions}
            />
          </div>
        ) : (
          <div style={{justifyContent: "center"}}>
            <img className="sign-one" src={oneSign} alt="sign-one" style={{height: "80px", marginLeft: "5px"}}/>
            <img className="sign-two" src={twoSign} alt="sign-two" style={{height: "80px"}}/>
            <img className="sign-three" src={threeSign} alt="sign-three" style={{height: "80px"}}/>
            {/* "Icons made by Freepik from www.flaticon.com" */}
            {(!this.state.imageFile) ? (
                <div>
                  <label htmlFor="file-upload" className="upload-file__button">    {/* Is what is visible*/}
                    Upload Image
                    <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={uploadIcon}/>
                  </label>
                  <input id="file-upload" className="upload-file__button"
                         type="file" accept="image/*"
                         onChange={this.getUploadedFileAsBinary}/>
                  {/* Is hidden via scss-file but contains the logic*/}
                </div>
            ) : <button type="button" className="upload-file__button" onClick={this.sendToAPI}>Translate!</button>
            }
                </div>
        )}
      </div>
    );
  }
}
