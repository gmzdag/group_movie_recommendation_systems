import React, { useRef } from 'react';

function UserNotFound({ username, onUpload, onCancel }) {
    const fileInputRef = useRef(null);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            onUpload(file, false);
        }
    };

    return (
        <div className="scenario-container fade-in">
            <div className="content-wrapper">
                <h1 className="scenario-title">
                    We don't have your films yet.
                </h1>

                <p className="scenario-subtitle">
                    Cinefuse needs your Letterboxd data to curate group recommendations.
                </p>

                <div className="instructions-box">
                    <ol className="instructions-list">
                        <li>Go to <a href="https://letterboxd.com/settings/data/" target="_blank" rel="noopener noreferrer" className="link-highlight">letterboxd.com/settings/data/</a></li>
                        <li>Scroll to "Export Your Data"</li>
                        <li>Click "Export"</li>
                        <li>Download the <code>letterboxd-{username || 'username'}-date-utc.zip</code> file</li>
                    </ol>
                </div>

                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept=".zip,.csv"
                    style={{ display: 'none' }}
                />

                <div className="button-group">
                    <button className="cta-button primary" onClick={handleUploadClick}>
                        Upload Letterboxd Data
                    </button>
                    <button className="cta-button secondary" onClick={onCancel}>
                        Cancel
                    </button>
                </div>

                <p className="error-microcopy">
                    Sorry — we can't recommend without your films.
                </p>
            </div>
        </div>
    );
}

export default UserNotFound;
