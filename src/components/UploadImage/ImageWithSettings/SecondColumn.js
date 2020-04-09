import React, { Component } from 'react';
import "../LoadingIcon.scss";
import smileyIcon from "../../../assets/icons/smiley-icon.png";
import resetIcon from "../../../assets/icons/reset-icon.png";
import "./SecondColumn.scss";
//import Environment from "../../../config/environment";
export default class SecondColumn extends Component {
    
    render() {
        return (
        <div>
            <div className="container">
                <div className="second-column">
                    <p> Your sign: </p> 
                    {/* <p>Listen to the corresponding verbal letter here: </p> */}
                    {/* {this.toAudio()}  */}
                </div>
                <div onClick={ () => window.location.reload()} style={{display: "flex", justifyContent:"center"}}><button className="restart__button">Upload another image! <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={resetIcon}/> </button></div>
            </div>
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