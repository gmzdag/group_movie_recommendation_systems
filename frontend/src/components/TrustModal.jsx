import React, { useState } from 'react';

function TrustModal({ isOpen, onClose, onTrust, onCustomPrompt }) {
    const [showPromptInput, setShowPromptInput] = useState(false);
    const [userPrompt, setUserPrompt] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [isTransitioning, setIsTransitioning] = useState(false);

    const handleTrust = () => {
        setIsProcessing(true);
        onTrust();
    };

    const handleNoTrust = () => {
        setIsTransitioning(true);
        setTimeout(() => {
            setShowPromptInput(true);
            setIsTransitioning(false);
        }, 300);
    };

    const handleSubmitPrompt = () => {
        if (!userPrompt.trim()) return;
        setIsProcessing(true);
        onCustomPrompt(userPrompt);
    };

    const handleBack = () => {
        setIsTransitioning(true);
        setTimeout(() => {
            setShowPromptInput(false);
            setUserPrompt('');
            setIsTransitioning(false);
        }, 300);
    };

    const handleModalClose = () => {
        setShowPromptInput(false);
        setUserPrompt('');
        setIsProcessing(false);
        setIsTransitioning(false);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={handleModalClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                {!showPromptInput ? (
                    <div className={`trust-question ${isTransitioning ? 'fade-out' : 'fade-in'}`}>
                        <h2 className="modal-title">Do you trust me?</h2>
                        <p className="modal-subtitle">
                            I can curate the perfect film for your group based on everyone's taste.
                        </p>

                        <div className="modal-buttons">
                            <button
                                className="modal-btn trust-btn"
                                onClick={handleTrust}
                                disabled={isProcessing || isTransitioning}
                            >
                                {isProcessing ? 'Curating...' : 'Yes, surprise me'}
                            </button>
                            <button
                                className="modal-btn custom-btn"
                                onClick={handleNoTrust}
                                disabled={isProcessing || isTransitioning}
                            >
                                I'll guide you
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className={`prompt-input-section ${isTransitioning ? 'fade-out' : 'fade-in'}`}>
                        <h2 className="modal-title">Tell me what you're looking for</h2>
                        <p className="modal-subtitle">
                            Describe the mood, genre, or vibe you want tonight.
                        </p>

                        <textarea
                            className="prompt-textarea"
                            value={userPrompt}
                            onChange={(e) => setUserPrompt(e.target.value)}
                            placeholder="e.g., Something dark and psychological, or a feel-good comedy..."
                            rows={4}
                            autoFocus
                        />

                        <div className="modal-buttons">
                            <button
                                className="modal-btn trust-btn"
                                onClick={handleSubmitPrompt}
                                disabled={isProcessing || !userPrompt.trim() || isTransitioning}
                            >
                                {isProcessing ? 'Analyzing...' : 'Find my film'}
                            </button>
                            <button
                                className="modal-btn secondary-btn"
                                onClick={handleBack}
                                disabled={isProcessing || isTransitioning}
                            >
                                Back
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default TrustModal;
