import React, { Component } from 'react';
import "../LoadingIcon.scss";
import smileyIcon from "../../../assets/icons/smiley-icon.png";
import resetIcon from "../../../assets/icons/reset-icon.png";
import "./SecondColumn.scss";
//import Environment from "../../../config/environment";
export default class SecondColumn extends Component {
    
    render() {
        return (
            <div className="second-column">
            {this.props.gettingAPIResponse 
                ? <div className="loading-icon__container" style={{paddingRight: "130px"}}><div className="lds-ellipsis"><div></div><div></div><div></div><div></div></div></div>
            : this.props.googleResponse === undefined 
            ? <div style={{width: "400px", marginLeft: "40px"}}>
                <p style={{textAlign: "center"}}>Press the button and we will analyze which <span style={{fontWeight: "bold"}}>sign you have given us</span> </p>
                    <div className="next-step__container">
                <button onClick={this.props.submitToCloudVisionAPI} className="next-step__button cta-button__hover"> Identify Sign <img height="20px" style={{marginLeft: "10px"}} alt="upload icon" src={smileyIcon}/></button>
            </div>    
            </div> 
            : <div> 
            <h4>The letter you signed is: </h4> 
            {/* <p>Listen to the corresponding verbal letter here: </p> */}
             {/* {this.toAudio()}  */}
            <div onClick={ () => window.location.reload()} style={{display: "flex", justifyContent:"center"}}><button className="restart__button">Upload another image! <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={resetIcon}/> </button></div>
                </div>}
            </div>
        )
    }
}

//  toAudio = () => {
    //      const textToSpeech = require('@google-cloud/text-to-speech');
    //      const fs = require('fs');
    //      const util = require('util');
    //      const path = require('path');

    //      const options = {
    //         keyFilename: path.join('../../../config/SSL-interpreter-bafad22fc797.json')
    //     }
    //     async function quickStart() {
            
    //         const client = new textToSpeech.TextToSpeechClient(options);
            
    //         // The text to synthesize
    //         const text = 'hello, world!';
    //         // Construct the request
    //         const request = {
    //           input: {text: text},
    //           // Select the language and SSML voice gender (optional)
    //           voice: {languageCode: 'sv-SE', ssmlGender: 'NEUTRAL'},
    //           // select the type of audio encoding
    //           audioConfig: {audioEncoding: 'MP3'},
    //         };
    //         // Performs the text-to-speech request
    //         const [response] = await client.synthesizeSpeech(request);
    //         // Write the binary audio content to a local file
    //         const writeFile = util.promisify(fs.writeFile);
    //         await writeFile('output.mp3', response.audioContent, 'binary');
    //         console.log('Audio content written to file: output.mp3');
    //       }
    //       quickStart();
    //      return(5)
    //  }