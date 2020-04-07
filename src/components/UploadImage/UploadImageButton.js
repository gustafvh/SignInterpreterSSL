import React, { Component } from 'react';
import "./UploadImageButton.scss";
import uploadIcon from "../../assets/icons/upload-icon.png";

export default class UploadImageButton extends Component {


    render() {
        return (
            <div>
            <div className="upload-file__container">
            <label htmlFor="file-upload" className="upload-file__button">    {/* Is what is visible*/}
                Select Image <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={uploadIcon}/>
            </label>


            <input id="file-upload" className="upload-file__button" type="file" accept="image/*" onChange={ (event) => this.props.uploadToFirebaseStorage(event) }/> {/* Is hidden via scss-file but contains the logic*/}
            </div>
            </div>
        )
    }
}
