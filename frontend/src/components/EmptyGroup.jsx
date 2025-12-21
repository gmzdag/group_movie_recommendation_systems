import React from 'react';

function EmptyGroup({ groupMembers, onAddMember }) {
    return (
        <div className="scenario-container fade-in">
            <div className="content-wrapper">
                {groupMembers.length === 0 ? (
                    <>
                        <h1 className="scenario-title">
                            An empty screen waits for its cast.
                        </h1>

                        <p className="scenario-subtitle">
                            Add Letterboxd profiles to begin the recommendation.
                        </p>

                        <button className="cta-button primary" onClick={onAddMember}>
                            Add Group Members
                        </button>
                    </>
                ) : (
                    <>
                        <h1 className="scenario-title">
                            Your Group
                        </h1>

                        <div className="members-list">
                            {groupMembers.map((member, idx) => (
                                <div key={idx} className="member-card">
                                    <span className="member-username">{member.username}</span>
                                </div>
                            ))}
                        </div>

                        <div className="button-group">
                            <button className="cta-button primary" onClick={onAddMember}>
                                Add Another Member
                            </button>
                            <button className="cta-button secondary">
                                Generate Recommendation
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

export default EmptyGroup;
