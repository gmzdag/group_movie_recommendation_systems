import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import TrustModal from '../components/TrustModal';

function GroupCreation() {
    const navigate = useNavigate();
    const [groupMembers, setGroupMembers] = useState([]);
    const [currentUsername, setCurrentUsername] = useState('');
    const [showUserNotFound, setShowUserNotFound] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [showSuccess, setShowSuccess] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const [importType, setImportType] = useState('');
    const [showError, setShowError] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [isUpdate, setIsUpdate] = useState(false);
    const fileInputRef = useRef(null);

    // Modal states
    const [showTrustModal, setShowTrustModal] = useState(false);

    const handleAddMember = async (e) => {
        e.preventDefault();
        if (!currentUsername.trim()) return;

        setIsLoading(true);
        setLoadingMessage('Checking user data…');

        try {
            const response = await fetch('http://127.0.0.1:8000/user_control/check_user', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: currentUsername.trim() })
            });

            const data = await response.json();

            // User exists and has data
            if (data.user_id !== null && data.has_data) {
                setGroupMembers([...groupMembers, {
                    username: currentUsername.trim(),
                    user_id: data.user_id
                }]);
                setCurrentUsername('');
                setIsLoading(false);
            } else {
                // User doesn't exist or has no data - needs to upload
                setIsLoading(false);
                setShowUserNotFound(true);
                setIsUpdate(false);
            }
        } catch (err) {
            console.error('Error checking user:', err);
            setIsLoading(false);
            setErrorMessage('Failed to check user. Please try again.');
            setShowError(true);
            setTimeout(() => setShowError(false), 3000);
        }
    };

    const handleUpdateMember = (username) => {
        setCurrentUsername(username);
        setIsUpdate(true);
        setShowUserNotFound(true);
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setShowUserNotFound(false);
        setIsLoading(true);
        setLoadingMessage('Rewatching your film history…');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('username', currentUsername);
        formData.append('update', isUpdate.toString());

        try {
            const response = await fetch('http://127.0.0.1:8000/user_control/upload_letterboxd', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                setIsLoading(false);
                setSuccessMessage(data.message || 'Your films are in!');
                setImportType(data.import_type || '');
                setShowSuccess(true);

                setTimeout(() => {
                    setShowSuccess(false);

                    if (isUpdate) {
                        // Just update the existing member
                        setGroupMembers(groupMembers.map(m =>
                            m.username === currentUsername ? { ...m, user_id: data.user_id } : m
                        ));
                    } else {
                        // Add new member
                        setGroupMembers([...groupMembers, {
                            username: currentUsername,
                            user_id: data.user_id
                        }]);
                    }

                    setCurrentUsername('');
                    setIsUpdate(false);
                    setSuccessMessage('');
                    setImportType('');
                }, 2500);
            } else {
                setIsLoading(false);
                setErrorMessage(data.detail || 'We opened the file, but couldn\'t find any films.');
                setShowError(true);
                setTimeout(() => {
                    setShowError(false);
                    setShowUserNotFound(true);
                }, 3000);
            }
        } catch (error) {
            console.error('Upload error:', error);
            setIsLoading(false);
            setErrorMessage('Upload failed. Please try again.');
            setShowError(true);
            setTimeout(() => {
                setShowError(false);
                setShowUserNotFound(true);
            }, 3000);
        }

        // Reset file input
        e.target.value = '';
    };

    const handleCancelUpload = () => {
        setShowUserNotFound(false);
        setCurrentUsername('');
        setIsUpdate(false);
    };

    const handleRemoveMember = (username) => {
        setGroupMembers(groupMembers.filter(m => m.username !== username));
    };

    const handleGenerateClick = () => {
        setShowTrustModal(true);
    };

    const handleTrustYes = async () => {
        setShowTrustModal(false);
        setIsLoading(true);
        setLoadingMessage('Generating your perfect recommendation...');

        try {
            const response = await fetch('http://127.0.0.1:8000/recommend/group', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_ids: groupMembers.map(m => m.user_id),
                    trust_mode: true,
                    top_k: 10
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate recommendation');
            }

            const data = await response.json();

            // Detailed logging for Trust Mode
            console.group('✨ Trust Mode - Top Recommendation');
            console.log('Method:', data.method);
            console.log('Total Candidates:', data.total_candidates);

            if (data.agent_recommendation) {
                console.group('🎬 Recommended Movie');
                console.log('Title:', data.agent_recommendation.title);
                console.log('Reason:', data.agent_recommendation.reason);
                console.log('Score:', data.agent_recommendation.score?.toFixed(2));
                console.log('Genres:', data.agent_recommendation.genres?.join(', '));
                console.log('Director:', data.agent_recommendation.director);
                console.log('Actors:', data.agent_recommendation.actors);
                console.log('Rating:', data.agent_recommendation.vote_average);
                console.log('Runtime:', data.agent_recommendation.runtime, 'min');
                console.log('Overview:', data.agent_recommendation.overview);
                console.groupEnd();
            }

            console.groupEnd();

            // TODO: Show recommendation in a results modal
            setIsLoading(false);
            alert(`🎬 ${data.agent_recommendation.title}\n\n${data.agent_recommendation.reason}`);

        } catch (error) {
            console.error('Recommendation error:', error);
            setIsLoading(false);
            setErrorMessage('Failed to generate recommendations. Please try again.');
            setShowError(true);
        }
    };

    const handleCustomPrompt = async (prompt) => {
        setShowTrustModal(false);
        setIsLoading(true);
        setLoadingMessage('Analyzing your preferences and filtering movies...');

        try {
            const response = await fetch('http://127.0.0.1:8000/recommend/group', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_ids: groupMembers.map(m => m.user_id),
                    trust_mode: false,
                    custom_prompt: prompt,
                    top_k: 10
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate recommendation');
            }

            const data = await response.json();

            // Detailed logging for AI Agent analysis
            console.group('🤖 AI Agent Analysis Results');

            if (data.filters_applied) {
                console.group('🔍 Sentiment Analysis & Filters Applied');
                console.log('User Prompt:', prompt);
                console.table(data.filters_applied);
                console.log('Total Candidates:', data.total_candidates);
                console.log('After Filtering:', data.filtered_count);
                console.groupEnd();
            }

            if (data.agent_recommendation) {
                console.group('⭐ Agent\'s Recommendation');
                console.log('Title:', data.agent_recommendation.title);
                console.log('Reason:', data.agent_recommendation.reason);
                console.log('Score:', data.agent_recommendation.score);
                console.log('Genres:', data.agent_recommendation.genres);
                console.log('Director:', data.agent_recommendation.director);
                console.log('Actors:', data.agent_recommendation.actors);
                console.log('Runtime:', data.agent_recommendation.runtime, 'min');
                console.log('Rating:', data.agent_recommendation.vote_average);
                console.log('Overview:', data.agent_recommendation.overview);
                console.groupEnd();
            }

            console.groupEnd();

            // TODO: Show recommendation in a results modal
            setIsLoading(false);
            alert(`🎬 ${data.agent_recommendation.title}\n\n${data.agent_recommendation.reason}`);

        } catch (error) {
            console.error('Recommendation error:', error);
            setIsLoading(false);
            setErrorMessage('Failed to generate recommendations. Please try again.');
            setShowError(true);
        }
    };

    return (
        <div className="cinefuse-app page-transition">
            <div className="scenario-container">
                <div className="content-wrapper">
                    {/* Loading State */}
                    {isLoading && (
                        <div className="overlay-state fade-in">
                            <div className="loading-spinner"></div>
                            <p className="loading-text">{loadingMessage}</p>
                        </div>
                    )}

                    {/* Success State */}
                    {showSuccess && (
                        <div className="overlay-state fade-in">
                            <div className="success-icon">✓</div>
                            <h1 className="scenario-title">{successMessage}</h1>
                            <p className="scenario-subtitle">
                                {importType === 'both' && "We're adding your ratings and watchlist to the group mix."}
                                {importType === 'ratings' && "We're adding your rated films to the group mix."}
                                {importType === 'watchlist' && "We're adding your watchlist to the group mix."}
                                {!importType && "We're adding them to the group mix."}
                            </p>
                        </div>
                    )}

                    {/* Error State */}
                    {showError && (
                        <div className="overlay-state fade-in">
                            <h1 className="scenario-title">{errorMessage}</h1>
                            <p className="scenario-subtitle">Make sure your export includes watched films, ratings, or diary data.</p>
                        </div>
                    )}

                    {/* User Not Found / Upload State */}
                    {showUserNotFound && !isLoading && !showSuccess && !showError && (
                        <div className="fade-in">
                            <h1 className="scenario-title">
                                {isUpdate ? 'Update your film data' : 'We don\'t have your films yet.'}
                            </h1>

                            <p className="scenario-subtitle">
                                {isUpdate
                                    ? 'Upload your latest Letterboxd export to update your recommendations.'
                                    : 'Cinefuse needs your Letterboxd data to curate group recommendations.'}
                            </p>

                            {!isUpdate && (
                                <div className="instructions-box">
                                    <ol className="instructions-list">
                                        <li>Go to <a href="https://letterboxd.com/settings/data/" target="_blank" rel="noopener noreferrer" className="link-highlight">letterboxd.com/settings/data/</a></li>
                                        <li>Scroll to "Export Your Data"</li>
                                        <li>Click "Export"</li>
                                        <li>Download the <code>letterboxd-{currentUsername || 'username'}-date-utc.zip</code> file</li>
                                        <li style={{ marginTop: '8px', fontSize: '0.9em', opacity: 0.8 }}>
                                            Or upload individual files: <code>ratings.csv</code> or <code>watchlist.csv</code>
                                        </li>
                                    </ol>
                                </div>
                            )}

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
                                <button className="cta-button secondary" onClick={handleCancelUpload}>
                                    Cancel
                                </button>
                            </div>

                            {!isUpdate && (
                                <p className="error-microcopy">
                                    Sorry — we can't recommend without your films.
                                </p>
                            )}
                        </div>
                    )}

                    {/* Main Group Creation Interface */}
                    {!showUserNotFound && !isLoading && !showSuccess && !showError && (
                        <div className="group-interface fade-in">
                            <h1 className="scenario-title">Create Your Group</h1>

                            <p className="scenario-subtitle">
                                Add Letterboxd profiles to begin the recommendation.
                            </p>

                            {/* Add Member Form */}
                            <form onSubmit={handleAddMember} className="member-form">
                                <div className="input-row">
                                    <input
                                        type="text"
                                        value={currentUsername}
                                        onChange={(e) => setCurrentUsername(e.target.value)}
                                        placeholder="letterboxd_username"
                                        className="username-input"
                                    />
                                    <button type="submit" className="add-btn" title="Add Member">
                                        +
                                    </button>
                                </div>
                            </form>

                            {/* Members List */}
                            {groupMembers.length > 0 && (
                                <div className="members-list">
                                    <h3 className="members-title">Group Members ({groupMembers.length})</h3>
                                    {groupMembers.map((member, idx) => (
                                        <div key={idx} className="member-card">
                                            <span className="member-username">@{member.username}</span>
                                            <div className="member-actions">
                                                <button
                                                    className="action-btn update-btn"
                                                    onClick={() => handleUpdateMember(member.username)}
                                                    title="Update data"
                                                >
                                                    ↻
                                                </button>
                                                <button
                                                    className="action-btn remove-btn"
                                                    onClick={() => handleRemoveMember(member.username)}
                                                    title="Remove member"
                                                >
                                                    ×
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Generate Recommendation Button */}
                            {groupMembers.length >= 2 && (
                                <button
                                    className="cta-button primary generate-btn"
                                    onClick={handleGenerateClick}
                                >
                                    Generate Group Recommendation
                                </button>
                            )}

                            {groupMembers.length === 1 && (
                                <p className="hint-text">Add at least one more member to generate recommendations</p>
                            )}

                            {/* Back Button */}
                            <button
                                className="back-link"
                                onClick={() => navigate('/')}
                            >
                                ← Back to Home
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Trust Modal */}
            <TrustModal
                isOpen={showTrustModal}
                onClose={() => setShowTrustModal(false)}
                onTrust={handleTrustYes}
                onCustomPrompt={handleCustomPrompt}
            />

            <div className="vignette-overlay"></div>
        </div>
    );
}

export default GroupCreation;
