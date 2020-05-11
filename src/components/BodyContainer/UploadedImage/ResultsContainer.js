import React, { Component } from 'react';
import "../LoadingIcon.scss";
import resetIcon from "../../../assets/icons/reset-icon.png";
import "./ResultsContainer.scss";
export default class ResultsContainer extends Component {

    renderSinglePrediction = (indexInPreds) => {
        return (
            <p className="single-prediction__container"><p className="preds-letter">{this.props.predictions[indexInPreds].letter} </p> &nbsp; <p style={{color: "#333", fontSize: "17px", textAlign: "center", margin: 0}}> with {this.props.predictions[indexInPreds].confidence}% confidence</p> </p>
        )
    }
    
    render() {
        return (
        <div>
                <div className="second-column">
                    <p className="second-column--title-text"> Your sign translates to the letter:</p>
                    {!this.props.loading &&
                    <div>
                        {this.renderSinglePrediction(0)}
                        {this.renderSinglePrediction(1)}
                        {this.renderSinglePrediction(2)}
                    </div>
                    }
                     {/*<p>Listen to the corresponding verbal letter here: </p>*/}
                     {/*{this.toAudio()}*/}
                </div>
                <div onClick={ () => window.location.reload()}>
                    <button className="upload-file__button">Upload another image <img style={{marginLeft: "10px"}} height="20px" alt="upload icon" src={resetIcon}/> </button></div>
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