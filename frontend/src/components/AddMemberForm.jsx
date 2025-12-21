import React, { useState } from 'react';

function AddMemberForm({ onSubmit, onCancel }) {
    const [username, setUsername] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (username.trim()) {
            onSubmit(username.trim());
        }
    };

    return (
        <div className="scenario-container fade-in">
            <div className="content-wrapper">
                <h1 className="scenario-title">
                    Add a Group Member
                </h1>

                <p className="scenario-subtitle">
                    Enter their Letterboxd username
                </p>

                <form onSubmit={handleSubmit} className="member-form">
                    <input
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="letterboxd_username"
                        className="username-input"
                        autoFocus
                    />

                    <div className="button-group">
                        <button type="submit" className="cta-button primary">
                            Add Member
                        </button>
                        <button type="button" className="cta-button secondary" onClick={onCancel}>
                            Cancel
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default AddMemberForm;
