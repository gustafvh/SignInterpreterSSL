import React from 'react';
import './Header.scss';

export default function Header() {
    return (
        <div className="header__container">
            <h3 className="header__small-title">Interpret</h3>
            <h1 className="header__big-title">Swedish Sign Language<span style={{color: "#0d98ba"}}>.</span></h1>
            <div style={{width: "400px", textAlign: "center"}}>
            <p>Upload your sign from the Swedish Sign Language hand alphabet and see its translation.</p>
            </div>
        </div>
    )
}

