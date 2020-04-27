import BodyContainer from './components/BodyContainer/BodyContainer.js';
import "./App.scss";
import Header from './components/Header/Header.js';
import Footer from './components/Footer/Footer.js';

import "normalize.css/normalize.css"; //NP, Resettar alla browsers default grejer
import 'dotenv';
import React, { Component } from 'react'
import firebase from "./firebase";
import Webcam from "react-webcam";

import cameraIcon from "./assets/icons/camera-icon.png";
import checkIcon from "./assets/icons/check-icon.png";
import crossIcon from "./assets/icons/cross-icon.png";




export default class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      currentStep: 1,
      gettingAPIResponse: false,
        webcamCapture: null,
        webcamCaptureBinary: null,
        webcamSelected: false,
        photoUsed: false
    };
  };

  startWebcam = () => {
    this.setState({
        webcamSelected: true
    })
  }

  encodeImageToBlob = (base64String) => {

        let byteString = atob(base64String.split(',')[1]);
        let byteStringLength = new ArrayBuffer(byteString.length);
        let intArray = new Uint8Array(byteStringLength);

        for (let i = 0; i < byteString.length; i++) {
            intArray[i] = byteString.charCodeAt(i);
        }
        return new Blob([intArray], { type: 'image/jpeg' });
    }

  takeSnaphotFromWebcam = () => {
      let screenshot = this.refs.webcamRef.getScreenshot();
      this.setState({webcamCapture: screenshot});
    }


    usePhoto = () => {
        this.setState({
            webcamCaptureBinary: this.encodeImageToBlob(this.state.webcamCapture),
            photoUsed: true
        })

   }

  renderWebcamCapture = () => {
        return (

            <div className="webcam-frame-container">
                <p className="header__big-title">Webcam</p>
                {this.state.webcamCapture ?
                    <div>
                    <img src={this.state.webcamCapture}/>
                    </div>
                    :
                <Webcam
                    audio={false}
                    height={450}
                    mirrored={false}
                    ref={'webcamRef'}
                    screenshotFormat="image/jpg"
                    width={800}
                />}
                <div className="buttons-container">
                    {!this.state.webcamCapture && <button className="upload-file__button" onClick={this.takeSnaphotFromWebcam}>Capture Photo  <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={cameraIcon}/></button>}
                    {this.state.webcamCapture && <button className="upload-file__button upload-file__button-red" onClick={()=> this.setState({webcamCapture: null})}>Retake Photo  <img style={{marginLeft: "10px"}} height="15px" alt="upload icon" src={crossIcon}/></button>}
                    {this.state.webcamCapture && <button className="upload-file__button" onClick={this.usePhoto}>Use this Photo  <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={checkIcon}/> </button>}
                </div>
            </div>
        );
    };



  

  render() {
    return (
        <div>
          <div className="app__container">

            {this.state.webcamSelected && !this.state.photoUsed ?
            this.renderWebcamCapture()
                : ( <div>
                <Header/>
            <BodyContainer webcamCapture={this.state.webcamCapture}
                           webcamCaptureBinary={this.state.webcamCaptureBinary}
                           startWebcam={this.startWebcam}/>
                </div>)}
                  <Footer currentStep={this.state.currentStep} />
          </div>
        </div>
    );
  }
}

