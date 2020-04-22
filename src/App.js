import BodyContainer from './components/BodyContainer/BodyContainer.js';
import "./App.scss";
import Header from './components/Header/Header.js';
import Footer from './components/Footer/Footer.js';

import "normalize.css/normalize.css"; //NP, Resettar alla browsers default grejer
import 'dotenv';
import React, { Component } from 'react'
import firebase from "./firebase";
import uploadIcon from "./assets/icons/upload-icon.png";



export default class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      currentStep: 1,
      gettingAPIResponse: false,
    };
  };

  changeCurrentStep = (stepToChangeTo) => {
    this.setState({
      currentStep: stepToChangeTo
    })
  }

  

  render() {
    return (
        <div>
          <div className="app__container">
            <Header/>
            <BodyContainer changeCurrentStep={this.changeCurrentStep}/>
            <Footer changeCurrentStep={this.changeCurrentStep} currentStep={this.state.currentStep} />
          </div>
        </div>
    );
  }
}

