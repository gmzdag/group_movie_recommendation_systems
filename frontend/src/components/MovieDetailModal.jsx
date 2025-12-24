import React from 'react';

const MovieDetailModal = ({ movie, onClose }) => {
    if (!movie) return null;

    const modelNames = {
        'H1': 'Dynamic Weighted Hybrid',
        'H2': 'Switching Hybrid',
        'H3': 'Watchlist-Enhanced Hybrid',
        'AI_AGENT': 'AI-Guided Selection'
    };

    const signalNames = {
        'COLLABORATIVE_FILTERING': 'Collaborative Filtering',
        'CONTENT_BASED': 'Content Similarity',
        'WATCHLIST': 'Watchlist Match',
        'HYBRID': 'Hybrid Signal',
        'AI_AGENT': 'AI Reasoning',
        'IBCF': 'Item-Based CF',
        'CBF': 'Content-Based'
    };

    return (
        <div
            className="modal-overlay"
            onClick={onClose}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'rgba(0,0,0,0.85)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1000,
                padding: '20px'
            }}
        >
            <div
                className="modal-content"
                onClick={(e) => e.stopPropagation()}
                style={{
                    background: 'linear-gradient(135deg, #1a1a1a, #2a2a2a)',
                    borderRadius: '24px',
                    padding: '40px',
                    maxWidth: '700px',
                    width: '100%',
                    maxHeight: '85vh',
                    overflowY: 'auto',
                    border: '1px solid rgba(255,255,255,0.1)',
                    boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                    position: 'relative'
                }}
            >
                {/* Close Button */}
                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '20px',
                        right: '20px',
                        background: 'rgba(255,255,255,0.1)',
                        border: 'none',
                        color: '#fff',
                        fontSize: '24px',
                        width: '40px',
                        height: '40px',
                        borderRadius: '50%',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}
                >
                    ×
                </button>

                {/* Title */}
                <h2 style={{ margin: '0 0 8px 0', fontSize: '1.8rem', color: '#fff' }}>
                    {movie.title}
                </h2>
                <p style={{ margin: '0 0 32px 0', color: '#888', fontSize: '0.9rem' }}>
                    Recommendation Details
                </p>

                {/* Model Info */}
                <div style={{ marginBottom: '32px' }}>
                    <h3 style={{ fontSize: '1.1rem', color: '#ffb74d', marginBottom: '12px' }}>
                        🤖 Recommendation Model
                    </h3>
                    <div style={{
                        background: 'rgba(255,183,77,0.1)',
                        padding: '16px',
                        borderRadius: '12px',
                        border: '1px solid rgba(255,183,77,0.2)'
                    }}>
                        <p style={{ margin: '0 0 8px 0', fontWeight: 'bold', color: '#ffb74d' }}>
                            {modelNames[movie.source_model] || movie.source_model}
                        </p>
                        <p style={{ margin: 0, fontSize: '0.9rem', color: '#ccc', lineHeight: '1.5' }}>
                            {movie.source_model === 'H1' && 'Dynamically balances collaborative filtering, content similarity, and watchlist preferences.'}
                            {movie.source_model === 'H2' && 'Intelligently switches between recommendation strategies based on user profiles.'}
                            {movie.source_model === 'H3' && 'Emphasizes items similar to movies in your watchlists.'}
                            {movie.source_model === 'AI_AGENT' && 'Uses advanced AI to understand your natural language preferences.'}
                        </p>
                    </div>
                </div>

                {/* Group Explanation */}
                <div style={{ marginBottom: '32px' }}>
                    <h3 style={{ fontSize: '1.1rem', color: '#81c784', marginBottom: '12px' }}>
                        💡 Why This Movie?
                    </h3>
                    <div style={{
                        background: 'rgba(129,199,132,0.1)',
                        padding: '16px',
                        borderRadius: '12px',
                        border: '1px solid rgba(129,199,132,0.2)'
                    }}>
                        <p style={{ margin: 0, fontSize: '0.95rem', color: '#ddd', lineHeight: '1.6' }}>
                            {movie.group_explanation || 'Recommended for the group.'}
                        </p>
                    </div>
                </div>

                {/* Per-User Reasons */}
                {movie.user_explanations && Object.keys(movie.user_explanations).length > 0 && (
                    <div>
                        <h3 style={{ fontSize: '1.1rem', color: '#ba68c8', marginBottom: '12px' }}>
                            👥 Individual Member Reasons
                        </h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {Object.entries(movie.user_explanations).map(([userId, expl]) => (
                                <div key={userId} style={{
                                    background: 'rgba(186,104,200,0.1)',
                                    padding: '12px 16px',
                                    borderRadius: '8px',
                                    border: '1px solid rgba(186,104,200,0.2)'
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                        <span style={{ fontSize: '0.85rem', color: '#ba68c8', fontWeight: 'bold' }}>
                                            Member {userId}
                                        </span>
                                        <span style={{ fontSize: '0.75rem', color: '#888', textTransform: 'uppercase' }}>
                                            {signalNames[expl.signal_source] || expl.signal_source}
                                        </span>
                                    </div>
                                    <p style={{ margin: 0, fontSize: '0.9rem', color: '#ccc' }}>
                                        {expl.primary_reason}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MovieDetailModal;
