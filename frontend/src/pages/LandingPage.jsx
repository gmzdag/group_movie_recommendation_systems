import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function LandingPage() {
    const [isHovered, setIsHovered] = useState(false);
    const navigate = useNavigate();

    const handleCreateGroup = () => {
        // Navigate with smooth transition
        navigate('/create-group');
    };

    return (
        <div className="cinefuse-app">
            <div className="hero-container">
                <div className="content-wrapper">
                    <h1 className="hero-title">
                        Every taste has <br />
                        <span className="italic-text">a role to play.</span>
                    </h1>

                    <div className="brand-text">Cinefuse</div>

                    <button
                        className="cta-button"
                        onMouseEnter={() => setIsHovered(true)}
                        onMouseLeave={() => setIsHovered(false)}
                        onClick={handleCreateGroup}
                    >
                        Create a Group Recommendation
                        <div className={`glow-effect ${isHovered ? 'active' : ''}`}></div>
                    </button>
                </div>
            </div>
            <div className="vignette-overlay"></div>
        </div>
    );
}

export default LandingPage;
