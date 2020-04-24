import BodyContainer from './components/BodyContainer/BodyContainer.js';
import "./App.scss";
import Header from './components/Header/Header.js';
import Footer from './components/Footer/Footer.js';

import "normalize.css/normalize.css"; //NP, Resettar alla browsers default grejer
import 'dotenv';
import React, { Component } from 'react'
import firebase from "./firebase";
import Webcam from "react-webcam";




export default class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      currentStep: 1,
      gettingAPIResponse: false,
        webcamCapture: null,
        webcamSelected: false
    };
  };

  startWebcam = () => {
    this.setState({
        webcamSelected: true
    })
  }

  takeSnaphotFromWebcam = () => {
        let screenshot = this.refs.webcamRef.getScreenshot();
        this.setState({webcamCapture: screenshot});
    }


    usePhoto = () => {
        this.setState({

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
                    height={500}
                    ref={'webcamRef'}
                    screenshotFormat="image/jpg"
                    width={800}
                />}
                <div className="buttons-container">
                <button className="upload-file__button" onClick={()=> this.setState({webcamCapture: null})}>Retake Photo</button>
                <button className="upload-file__button" onClick={this.takeSnaphotFromWebcam}>Capture Photo</button>
                <button className="upload-file__button" onClick={this.usePhoto}>Use this Photo</button>
                </div>
            </div>
        );
    };



  

  render() {
    return (
        <div>
          <div className="app__container">

            {this.state.webcamSelected ?
            this.renderWebcamCapture()
                : ( <div>
                <Header/>
            <BodyContainer webcamCapture={this.state.webcamCapture} startWebcam={this.startWebcam}/>
                </div>)}
                  <Footer currentStep={this.state.currentStep} />
          </div>
        </div>
    );
  }
}

