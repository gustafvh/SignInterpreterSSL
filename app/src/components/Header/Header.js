import React from 'react';
import './Header.scss';

export default function Header() {
    return (
        <div className="header__container">
            <h3 className="header__small-title">Interpret </h3>
            <h1 className="header__big-title">Sign Language<span style={{color: "#186cb5"}}>.</span></h1>
            <p href="https//halco.se">Upload your sign and see its translation.</p>
        </div>
    )
}

