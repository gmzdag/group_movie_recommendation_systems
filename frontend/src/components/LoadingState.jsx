import React from 'react';

function LoadingState({ message = "Scanning filmographies…" }) {
    return (
        <div className="scenario-container fade-in">
            <div className="content-wrapper">
                <div className="loading-spinner"></div>

                <p className="loading-text">
                    {message}
                </p>
            </div>
        </div>
    );
}

export default LoadingState;
