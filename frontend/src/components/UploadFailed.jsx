import React from 'react';

function UploadFailed({ onRetry }) {
    return (
        <div className="scenario-container fade-in">
            <div className="content-wrapper">
                <h1 className="scenario-title">
                    We opened the file, but couldn't find any films.
                </h1>

                <p className="scenario-subtitle">
                    Make sure your export includes watched films, ratings, or diary data.
                </p>

                <button className="cta-button primary" onClick={onRetry}>
                    Try Again
                </button>
            </div>
        </div>
    );
}

export default UploadFailed;
