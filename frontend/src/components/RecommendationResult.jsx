import React, { useState } from 'react';

function RecommendationResult({ isOpen, onClose, recommendation, isLoading }) {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content recommendation-modal" onClick={(e) => e.stopPropagation()}>
                {isLoading ? (
                    <div className="loading-state fade-in">
                        <div className="loading-spinner"></div>
                        <p className="loading-text">Curating your perfect film…</p>
                    </div>
                ) : recommendation ? (
                    <div className="recommendation-content fade-in">
                        <h2 className="modal-title">Tonight's Recommendation</h2>

                        <div className="movie-card">
                            <h3 className="movie-title">{recommendation.title}</h3>
                            {recommendation.year && (
                                <span className="movie-year">({recommendation.year})</span>
                            )}

                            {recommendation.genres && (
                                <div className="movie-genres">
                                    {recommendation.genres.map((genre, idx) => (
                                        <span key={idx} className="genre-tag">{genre}</span>
                                    ))}
                                </div>
                            )}

                            {recommendation.reason && (
                                <p className="recommendation-reason">{recommendation.reason}</p>
                            )}

                            {recommendation.score && (
                                <div className="recommendation-score">
                                    <span className="score-label">Match Score:</span>
                                    <span className="score-value">{(recommendation.score * 100).toFixed(0)}%</span>
                                </div>
                            )}
                        </div>

                        <div className="modal-buttons">
                            <button className="modal-btn trust-btn" onClick={onClose}>
                                Perfect choice
                            </button>
                            <button className="modal-btn secondary-btn" onClick={() => window.location.reload()}>
                                Try another group
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="error-state fade-in">
                        <h2 className="modal-title">Couldn't find a match</h2>
                        <p className="modal-subtitle">Try adjusting your group or preferences.</p>
                        <button className="modal-btn secondary-btn" onClick={onClose}>
                            Close
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

export default RecommendationResult;
