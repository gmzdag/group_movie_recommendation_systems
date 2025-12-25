import React from 'react';

const MovieDetailModal = ({ movie, userMap = {}, onClose }) => {
    if (!movie) return null;

    // Helper to replace user IDs with usernames in text
    const cleanText = (text) => {
        if (!text) return text;
        let cleaned = text;
        Object.entries(userMap).forEach(([id, username]) => {
            const regex = new RegExp(id, 'g');
            cleaned = cleaned.replace(regex, `@${username} `);
        });
        return cleaned;
    };

    const modelNames = {
        'H1': 'Item-Based Hybrid (IBCF + Content)',
        'H2': 'User-Based Hybrid (UBCF + Content)',
        'H3': 'Watchlist Content Filter',
        'AI_AGENT': 'AI-Guided Selection'
    };

    const modelDescriptions = {
        'H1': 'Recommends movies similar to ones you\'ve enjoyed, combining item similarity with content features.',
        'H2': 'Finds users with similar taste and recommends their favorites, enhanced with content analysis.',
        'H3': 'Discovers new films similar to your watchlist using content-based filtering.',
        'AI_AGENT': 'AI-curated selection based on your specific request and group preferences.'
    };

    const signalNames = {
        'COLLABORATIVE_FILTERING': 'Collaborative Filtering',
        'CONTENT_BASED': 'Content Similarity',
        'WATCHLIST': 'Watchlist Match',
        'WATCHLIST_INSPIRED': 'Watchlist-Inspired',
        'COLLABORATIVE': 'Item Similarity',
        'ADAPTIVE': 'User Taste',
        'HYBRID': 'Hybrid Signal',
        'AI_AGENT': 'AI Reasoning',
        'IBCF': 'Item-Based CF',
        'CBF': 'Content-Based',
        'UBCF': 'User-Based CF'
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
                    position: 'relative',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    boxShadow: '0 30px 60px rgba(0,0,0,0.8)'
                }}
            >
                {/* Close Button */}
                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '24px',
                        right: '24px',
                        background: 'none',
                        border: 'none',
                        color: '#666',
                        fontSize: '2rem',
                        cursor: 'pointer',
                        lineHeight: '1',
                        padding: '4px',
                        transition: 'color 0.2s'
                    }}
                    onMouseEnter={(e) => e.target.style.color = '#fff'}
                    onMouseLeave={(e) => e.target.style.color = '#666'}
                >
                    ×
                </button>

                {/* Title */}
                <h2 style={{ margin: '0 0 4px 0', fontSize: '2.2rem', color: '#fff', fontFamily: 'var(--font-serif)', fontWeight: '300', letterSpacing: '-0.01em', lineHeight: '1.2' }}>
                    {movie.title}
                </h2>
                <p style={{ margin: '0 0 40px 0', color: '#666', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', fontWeight: '300', letterSpacing: '0.05em' }}>
                    Recommendation Details
                </p>

                {/* Model Info */}
                <div style={{ marginBottom: '40px' }}>
                    <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>
                        Recommendation Model
                    </h3>
                    <div style={{
                        background: 'rgba(255, 255, 255, 0.02)',
                        padding: '24px',
                        borderRadius: '8px',
                        border: '1px solid rgba(255, 255, 255, 0.06)'
                    }}>
                        <p style={{ margin: '0 0 12px 0', fontWeight: '400', color: '#e0e0e0', fontFamily: 'var(--font-sans)', fontSize: '1.05rem', letterSpacing: '-0.01em' }}>
                            {modelNames[movie.source_model] || movie.source_model}
                        </p>
                        <p style={{ margin: 0, fontSize: '0.9rem', color: '#999', lineHeight: '1.7', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>
                            {modelDescriptions[movie.source_model] || 'Advanced hybrid recommendation algorithm.'}
                        </p>
                    </div>
                </div>

                {/* Why This Film */}
                <div style={{ marginBottom: '40px' }}>
                    <h3 style={{ fontSize: '0.7rem', color: '#FF3D3D', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '500' }}>
                        Why This Film
                    </h3>
                    <div style={{
                        background: 'rgba(255, 61, 61, 0.08)',
                        padding: '24px',
                        borderRadius: '8px',
                        border: '1px solid rgba(255, 61, 61, 0.15)',
                        borderLeft: '2px solid rgba(255, 61, 61, 0.3)'
                    }}>
                        <p style={{ margin: 0, fontSize: '0.95rem', color: '#c0c0c0', lineHeight: '1.8', fontFamily: 'var(--font-sans)', fontWeight: '300', fontStyle: 'italic' }}>
                            {cleanText(movie.group_explanation) || 'Recommended for the group.'}
                        </p>
                    </div>
                </div>

                {/* Per-User Reasons */}
                {movie.user_explanations && Object.keys(movie.user_explanations).length > 0 && (
                    <div>
                        <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>
                            Individual Reasons
                        </h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {Object.entries(movie.user_explanations).map(([userId, expl]) => (
                                <div key={userId} style={{
                                    background: 'rgba(255, 255, 255, 0.02)',
                                    padding: '16px 20px',
                                    borderRadius: '6px',
                                    border: '1px solid rgba(255, 255, 255, 0.06)'
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                        <span style={{ fontSize: '0.75rem', color: '#aaa', fontWeight: '400', fontFamily: 'var(--font-sans)', letterSpacing: '0.05em' }}>
                                            {userMap[userId] ? `@${userMap[userId]} ` : `Member ${userId} `}
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
