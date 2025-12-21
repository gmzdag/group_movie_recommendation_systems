import React from 'react';

function UploadSuccess() {
    return (
        <div className="scenario-container fade-in">
            <div className="content-wrapper">
                <div className="success-icon">✓</div>

                <h1 className="scenario-title">
                    Your films are in.
                </h1>

                <p className="scenario-subtitle">
                    We're adding them to the group mix.
                </p>
            </div>
        </div>
    );
}

export default UploadSuccess;
