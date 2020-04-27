import React, { Component } from 'react';
import "./UploadedImage.scss";
import ResultsContainer from './ResultsContainer.js';

export default class UploadedImage extends Component {
    render() {
        return (
        <div>
            <div className="container">
                <div className="first-column">
                    <img className="image" src={this.props.imageFileObjectURL ? this.props.imageFileObjectURL : this.props.webcamCapture} alt=""/>
                    <p className="filename-image-text"> Image Uploaded: {this.props.imageFile ? this.props.imageFile.name : "Webcam Image"}  </p>
                </div>
                <ResultsContainer predictions={this.props.predictions}/>
            </div>
        </div>

        
        )
    }
}
