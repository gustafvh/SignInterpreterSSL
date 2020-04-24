import React, { Component } from 'react';
import "./UploadedImage.scss";
import ResultsContainer from './ResultsContainer.js';

export default class UploadedImage extends Component {
    render() {
        return (
        <div>
            <div className="container">
                <div className="first-column">
                    <img className="image" src={this.props.imageFileObjectURL} alt=""/>
                    <p className="filename-image-text"> Image Uploaded: {this.props.imageFile.name} </p>
                </div>
                <ResultsContainer predictions={this.props.predictions}/>
            </div>
        </div>

        
        )
    }
}
