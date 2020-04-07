import React, { Component } from 'react';
import "../LoadingIcon.scss";
import smileyIcon from "../../../assets/icons/smiley-icon.png";
import resetIcon from "../../../assets/icons/reset-icon.png";
import "./SecondColumn.scss";

export default class SecondColumn extends Component {

    renderSimilairImages = (arrayWithUrls) => {
        return (
        <div className="rendered-images__container">
                {arrayWithUrls.map((singleImageUrl) => 
                    <img className="rendered-images__single-image" onError={e => e.target.style.display = 'none'} src={singleImageUrl} alt={singleImageUrl}/>
                    )}
            </div> );
    }

    render() {
        return (
            <div className="second-column">
            
            {this.props.gettingAPIResponse 
                ? <div className="loading-icon__container"><div className="lds-ellipsis"><div></div><div></div><div></div><div></div></div></div>
            : this.props.googleResponse === undefined 
            ? <div style={{width: "400px"}}>
                <p style={{textAlign: "center"}}>Press the button and we will analyze which <span style={{fontWeight: "bold"}}>sign you have given us</span> </p>
                    <div className="next-step__container">
                <button onClick={this.props.submitToCloudVisionAPI} className="next-step__button cta-button__hover"> Identify Sign <img height="20px" style={{marginLeft: "10px"}} alt="upload icon" src={smileyIcon}/></button>
            </div>    
            </div> 
            : <div> 
            <h4>Your sign is: </h4>
            <p>We labeled the image as <span style={{fontWeight: "bold"}}>{this.props.imageLabels[0]}</span>.</p>
            {this.renderSimilairImages(this.props.similairImagesUrls)}
            <p className="small-information-text">Scroll down for more images.</p>
            <div onClick={ () => window.location.reload()} style={{display: "flex", justifyContent:"center"}}><button className="restart__button">Do it again! <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={resetIcon}/> </button></div>
                </div>}
            </div>
        )
    }
}
