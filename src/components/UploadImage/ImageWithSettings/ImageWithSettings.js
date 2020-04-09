import React, { Component } from 'react';
import "./ImageWithSettings.scss";
import SecondColumn from './SecondColumn.js';

export default class ImageWithSettings extends Component {
    render() {
        return (
        <div>
            <div className="container">
                <div className="first-column">
                    <img className="image" src={this.props.uploadedFileUrl} alt=""/>
                    <p> Your image </p> 
                </div>
                <SecondColumn submitToCloudVisionAPI={this.props.submitToCloudVisionAPI} gettingAPIResponse={this.props.gettingAPIResponse} googleResponse={this.props.googleResponse} similairImagesUrls={this.props.similairImagesUrls} imageLabels={this.props.imageLabels}/>
            </div>
        </div>

        
        )
    }
}
