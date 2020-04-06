import "./App.scss";
import Header from './Header/Header.js';
import Footer from './Footer/Footer.js';

import "normalize.css/normalize.css"; //NP, Reset all browsers default things
import React, {Component} from 'react'

export default class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      currentStep: 1
    };
  };

  render() {
  return (
    <div>
      <div className="app__container">
        <Header/>
        <p> Hello </p>
        <Footer/>
      </div>
    </div>
  );
}
}

