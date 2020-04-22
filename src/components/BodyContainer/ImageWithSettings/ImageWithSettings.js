import React, { Component } from 'react';
import "./ImageWithSettings.scss";
import SecondColumn from './SecondColumn.js';

export default class ImageWithSettings extends Component {
    render() {
        return (
        <div>
            <div className="container">
                <div className="first-column">
                    <img className="image" src={this.props.imageFileObjectURL} alt=""/>
                    <p> Your image: {this.props.imageFile.name} </p>
                </div>
                <SecondColumn predictions={this.props.predictions}/>
            </div>
        </div>

        
        )
    }
}
