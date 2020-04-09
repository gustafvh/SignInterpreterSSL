import React, {Component} from 'react';
import "./Footer.scss";


export default class Footer extends Component {
    render() {
        return (
            <div className="footer__container">

            <p className="footer_small-text"> Made by Gustaf Halvardsson & Johanna Peterson |
            <a className="footer_small-text" href="https://github.com/gustafvh/SignInterpreterSSL"> Privacy Policy |</a> 
            <a className="footer_small-text" href="https://github.com/gustafvh/SignInterpreterSSL"> GitHub</a>
            </p>
            
        </div>
        )
    }
}

