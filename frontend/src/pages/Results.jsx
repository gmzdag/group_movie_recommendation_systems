import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MovieDetailModal from '../components/MovieDetailModal';

function Results() {
    const location = useLocation();
    const navigate = useNavigate();
    const { results, userPrompt, mode } = location.state || {};
    const [selectedMovie, setSelectedMovie] = useState(null);
    const [trailerUrl, setTrailerUrl] = useState(null);

    // DEBUG: Log results
    console.log("=== RESULTS PAGE DEBUG ===");
    console.log("Full location.state:", location.state);
    console.log("Results object:", results);
    if (results) {
        console.log("Section A:", results.section_a_top_recommendations);
        console.log("Section A length:", results.section_a_top_recommendations?.length);
        console.log("Section B:", results.section_b_common_watchlist);
        console.log("Section C:", results.section_c_shared_interests);
    }

    if (!results) {
        return (
            <div className="cinefuse-app">
                <div className="content-wrapper" style={{ textAlign: 'center', paddingTop: '100px' }}>
                    <h1 className="scenario-title">No Results Found</h1>
                    <button className="cta-button primary" onClick={() => navigate('/')}>
                        Go Home
                    </button>
                </div>
            </div>
        );
    }

    const {
        section_a_top_recommendations: sectionA,
        section_b_common_watchlist: sectionB,
        section_c_shared_interests: sectionC
    } = results;

    // DEBUG: Log after destructuring
    console.log("After destructuring - sectionA:", sectionA);
    console.log("sectionA is array?", Array.isArray(sectionA));
    console.log("sectionA length:", sectionA?.length);

    return (
        <div className="cinefuse-app page-transition">
            <div className="results-container" style={{ padding: '0', maxWidth: '100%', margin: '0' }}>

                {/* HEADER REMOVED FOR DESKTOP HERO DESIGN */}

                {/* TOP PICK HERO SECTION (Rank #1) - Full Screen Desktop Style */}
                {sectionA && sectionA.length > 0 && (() => {
                    const topMovie = sectionA[0];
                    return (
                        <div className="hero-recommendation-desktop" style={{
                            position: 'relative',
                            width: '100vw',
                            height: '85vh',
                            marginLeft: 'calc(-50vw + 50%)', // Break out of container
                            marginRight: 'calc(-50vw + 50%)',
                            overflow: 'hidden',
                            marginBottom: '60px',
                            boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
                        }}>
                            {/* Background Image */}
                            <div style={{
                                position: 'absolute',
                                top: 0, left: 0, right: 0, bottom: 0,
                                backgroundImage: `url(${topMovie.backdrop_url || topMovie.poster_url})`,
                                backgroundSize: 'cover',
                                backgroundPosition: 'center top',
                                filter: 'brightness(0.5) saturate(1.1)'
                            }} />

                            {/* Gradient Overlay */}
                            <div style={{
                                position: 'absolute',
                                top: 0, left: 0, right: 0, bottom: 0,
                                background: 'linear-gradient(to right, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.5) 60%, transparent 100%)',
                                zIndex: 1
                            }} />

                            {/* Content Container */}
                            <div style={{
                                position: 'relative',
                                padding: '0 5%',
                                maxWidth: '100%',
                                margin: '0 auto',
                                height: '100%',
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'center',
                                alignItems: 'flex-end', // Right align
                                textAlign: 'right', // Right align text
                                zIndex: 2
                            }}>
                                {/* Rank Badge */}
                                <div className="rank-badge" style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '10px',
                                    background: 'rgba(255, 61, 61, 0.2)',
                                    border: '1px solid #FF3D3D',
                                    color: '#FF3D3D',
                                    padding: '8px 20px',
                                    borderRadius: '50px',
                                    fontSize: '1rem',
                                    fontWeight: '700',
                                    marginBottom: '24px',
                                    width: 'fit-content',
                                    backdropFilter: 'blur(5px)'
                                }}>
                                    <span>🏆</span> #1 GROUP PICK
                                </div>

                                {/* Title - Larger */}
                                <h1 style={{
                                    fontSize: '7rem', // Increased size
                                    fontWeight: '900',
                                    lineHeight: '1',
                                    marginBottom: '20px',
                                    color: '#fff',
                                    maxWidth: '1200px',
                                    textShadow: '0 4px 20px rgba(0,0,0,0.5)'
                                }}>
                                    {topMovie.title}
                                </h1>

                                {/* Metadata */}
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '24px', marginBottom: '32px', fontSize: '1.4rem', color: '#ddd' }}>
                                    <span style={{
                                        color: '#FF3D3D',
                                        fontWeight: '800',
                                        fontSize: '1.6rem'
                                    }}>
                                        {topMovie.group_score > 1 ? Math.round((topMovie.group_score / 5) * 100) : Math.round(topMovie.group_score * 100)}% Match
                                    </span>
                                    {topMovie.genres && (
                                        <>
                                            <span style={{ opacity: 0.5 }}>|</span>
                                            <span>{topMovie.genres.replace(/\|/g, ', ')}</span>
                                        </>
                                    )}
                                </div>

                                {/* Overview / Plot (New addition) */}
                                {/* Overview / Plot */}
                                <div style={{ marginBottom: '32px', maxWidth: '800px' }}>
                                    <h3 style={{ fontSize: '1.2rem', color: '#FF3D3D', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '2px' }}>Plot</h3>
                                    <p style={{
                                        fontSize: '1.4rem',
                                        lineHeight: '1.6',
                                        color: '#e0e0e0',
                                        textShadow: '0 2px 4px rgba(0,0,0,0.5)'
                                    }}>
                                        {topMovie.overview || "No plot overview available."}
                                    </p>
                                </div>

                                {/* AI Explanation */}
                                {topMovie.group_explanation && (
                                    <div style={{ marginBottom: '48px', maxWidth: '800px', background: 'rgba(255, 61, 61, 0.1)', padding: '20px', borderRadius: '12px', borderRight: '4px solid #FF3D3D' }}>
                                        <h3 style={{ fontSize: '1rem', color: '#FF3D3D', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>Warum dieser Film?</h3>
                                        <p style={{ fontSize: '1.1rem', fontStyle: 'italic', color: '#ccc', margin: 0 }}>
                                            "{topMovie.group_explanation}"
                                        </p>
                                    </div>
                                )}

                                {/* Action Buttons */}
                                <div style={{ display: 'flex', gap: '20px' }}>
                                    {topMovie.trailer_url && (
                                        <button
                                            onClick={() => setTrailerUrl(topMovie.trailer_url)}
                                            style={{
                                                background: '#FF3D3D',
                                                color: 'white',
                                                border: 'none',
                                                padding: '18px 40px',
                                                borderRadius: '8px',
                                                fontSize: '1.1rem',
                                                fontWeight: '700',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '12px',
                                                transition: 'all 0.2s',
                                                boxShadow: '0 10px 30px rgba(255, 61, 61, 0.3)'
                                            }}
                                            onMouseEnter={(e) => {
                                                e.target.style.transform = 'translateY(-2px)';
                                                e.target.style.boxShadow = '0 15px 40px rgba(255, 61, 61, 0.5)';
                                            }}
                                            onMouseLeave={(e) => {
                                                e.target.style.transform = 'translateY(0)';
                                                e.target.style.boxShadow = '0 10px 30px rgba(255, 61, 61, 0.3)';
                                            }}
                                        >
                                            ▶ Watch Trailer
                                        </button>
                                    )}
                                    <button
                                        onClick={() => setSelectedMovie(topMovie)}
                                        style={{
                                            background: 'rgba(255,255,255,0.1)',
                                            color: 'white',
                                            border: '2px solid rgba(255,255,255,0.3)',
                                            padding: '18px 40px',
                                            borderRadius: '8px',
                                            fontSize: '1.1rem',
                                            fontWeight: '700',
                                            cursor: 'pointer',
                                            backdropFilter: 'blur(10px)',
                                            transition: 'all 0.2s'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.target.style.background = 'rgba(255,255,255,0.2)';
                                            e.target.style.borderColor = 'white';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.background = 'rgba(255,255,255,0.1)';
                                            e.target.style.borderColor = 'rgba(255,255,255,0.3)';
                                        }}
                                    >
                                        More Details
                                    </button>
                                </div>

                                {/* AI Explanation Mini-Badge */}
                                {/* AI Reason removed from bottom since it's now explicit above */}
                            </div>
                        </div>
                    );
                })()}

                <section className="results-section fade-in delay-1" style={{ marginBottom: '80px' }}>
                    <div className="section-header" style={{ marginBottom: '32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div>
                            <h3 className="section-title" style={{ fontSize: '1.8rem', color: '#fff', marginBottom: '8px' }}>Detailed Breakdown</h3>
                            <p className="section-desc" style={{ color: '#888', margin: 0 }}>Next top recommendations</p>
                        </div>
                    </div>

                    {/* GRID SECTION (Rank #2-10) - Full Width */}
                    <div className="cards-grid" style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', // Wider cards
                        gap: '24px',
                        padding: '0 40px' // Add side padding since container is full width
                    }}>
                        {sectionA && sectionA.slice(1).map((movie, idx) => {
                            // Determine actual rank (index + 2 because we skipped the first one)
                            const rank = idx + 2;

                            return (
                                <div key={movie.movie_id || idx} className="movie-card-poster" style={{
                                    position: 'relative',
                                    borderRadius: '16px',
                                    overflow: 'hidden',
                                    background: '#1a1a1a',
                                    border: '1px solid rgba(255,255,255,0.05)',
                                    transition: 'all 0.3s ease',
                                    cursor: 'pointer',
                                    aspectRatio: '2/3'
                                }}
                                    onClick={() => setSelectedMovie(movie)}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.transform = 'translateY(-8px)';
                                        e.currentTarget.querySelector('.card-overlay').style.opacity = '1';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.transform = 'translateY(0)';
                                        e.currentTarget.querySelector('.card-overlay').style.opacity = '0';
                                    }}>
                                    {/* Poster Image */}
                                    <img
                                        src={movie.poster_url}
                                        alt={movie.title}
                                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                        onError={(e) => e.target.style.display = 'none'} // Very basic fallback
                                    />
                                    {!movie.poster_url && (
                                        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444' }}>
                                            No Poster
                                        </div>
                                    )}

                                    {/* Gradient Overlay (Always visible at bottom for text readability) */}
                                    <div style={{
                                        position: 'absolute',
                                        left: 0, right: 0, bottom: 0,
                                        height: '50%',
                                        background: 'linear-gradient(to top, rgba(0,0,0,0.9), transparent)',
                                        pointerEvents: 'none'
                                    }} />

                                    {/* Rank Badge */}
                                    <div style={{
                                        position: 'absolute',
                                        top: '12px',
                                        left: '12px',
                                        background: 'rgba(0,0,0,0.6)',
                                        backdropFilter: 'blur(4px)',
                                        color: '#fff',
                                        padding: '4px 10px',
                                        borderRadius: '8px',
                                        fontSize: '0.8rem',
                                        fontWeight: 'bold',
                                        border: '1px solid rgba(255,255,255,0.1)'
                                    }}>
                                        #{rank}
                                    </div>

                                    {/* Info Content */}
                                    <div style={{
                                        position: 'absolute',
                                        bottom: 0, left: 0, right: 0,
                                        padding: '16px'
                                    }}>
                                        <h4 style={{ margin: '0 0 4px 0', fontSize: '1rem', lineHeight: '1.3' }}>{movie.title}</h4>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem' }}>
                                            <span style={{ color: '#ccc' }}>{movie.release_date?.split('-')[0]}</span>
                                            <span style={{ color: '#FF3D3D', fontWeight: 'bold' }}>
                                                {movie.group_score > 1 ? Math.round((movie.group_score / 5) * 100) : Math.round(movie.group_score * 100)}%
                                            </span>
                                        </div>
                                    </div>

                                    {/* Interactive Overlay */}
                                    <div className="card-overlay" style={{
                                        position: 'absolute',
                                        top: 0, left: 0, right: 0, bottom: 0,
                                        background: 'rgba(0,0,0,0.7)',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        opacity: 0,
                                        transition: 'opacity 0.2s ease',
                                        backdropFilter: 'blur(2px)',
                                        gap: '12px'
                                    }}>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setSelectedMovie(movie);
                                            }}
                                            style={{
                                                padding: '10px 20px',
                                                borderRadius: '20px',
                                                background: 'white',
                                                color: 'black',
                                                border: 'none',
                                                fontWeight: 'bold',
                                                cursor: 'pointer'
                                            }}
                                        >
                                            View Details
                                        </button>
                                        {movie.trailer_url && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setTrailerUrl(movie.trailer_url);
                                                }}
                                                style={{
                                                    padding: '10px 20px',
                                                    borderRadius: '20px',
                                                    background: 'rgba(255, 61, 61, 0.2)',
                                                    border: '1px solid #FF3D3D',
                                                    color: 'white',
                                                    fontWeight: 'bold',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                ▶ Trailer
                                            </button>
                                        )}
                                        <p style={{ margin: '16px 20px 0', fontSize: '0.8rem', textAlign: 'center', fontStyle: 'italic', color: '#ccc' }}>
                                            "{movie.group_explanation && movie.group_explanation.length > 60 ? movie.group_explanation.substring(0, 60) + '...' : movie.group_explanation}"
                                        </p>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </section>

                {/* SECTION B: COMMON WATCHLIST */}
                {sectionB && sectionB.length > 0 && (
                    <section className="results-section fade-in delay-2" style={{ marginTop: '80px' }}>
                        <div className="section-header" style={{ marginBottom: '32px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '16px' }}>
                            <h2 className="section-title" style={{ fontSize: '2rem', color: '#fff' }}>From Watchlists</h2>
                            <p className="section-desc" style={{ color: '#888' }}>Movies you already want to see</p>
                        </div>

                        {/* Horizontal Scroll / Slider Layout for Watchlist */}
                        {/* Horizontal Scroll / Slider Layout for Watchlist */}
                        <div className="watchlist-slider hide-scrollbar" style={{
                            display: 'flex',
                            overflowX: 'auto',
                            gap: '24px',
                            padding: '10px 40px',
                            scrollBehavior: 'smooth'
                        }}>
                            <style>{`
                                .hide-scrollbar::-webkit-scrollbar {
                                    display: none;
                                }
                                .hide-scrollbar {
                                    -ms-overflow-style: none;
                                    scrollbar-width: none;
                                }
                            `}</style>
                            {sectionB.map((item) => (
                                <div key={item.movie_id} className="watchlist-card-poster" style={{
                                    flex: '0 0 220px', // Fixed width for slider items
                                    position: 'relative',
                                    borderRadius: '16px',
                                    overflow: 'hidden',
                                    aspectRatio: '2/3',
                                    background: '#1a1a1a',
                                    border: '1px solid rgba(255,255,255,0.05)',
                                    cursor: 'pointer',
                                    transition: 'transform 0.3s ease'
                                }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.transform = 'scale(1.05)';
                                        e.currentTarget.querySelector('.wl-overlay').style.opacity = '1';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.transform = 'scale(1)';
                                        e.currentTarget.querySelector('.wl-overlay').style.opacity = '0';
                                    }}>
                                    {/* Full Poster */}
                                    <img
                                        src={item.poster_url}
                                        alt={item.title}
                                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                        onError={(e) => e.target.style.display = 'none'}
                                    />
                                    {!item.poster_url && (
                                        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444' }}>
                                            {item.title}
                                        </div>
                                    )}

                                    {/* Overlay on Hover */}
                                    <div className="wl-overlay" style={{
                                        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                                        background: 'rgba(0,0,0,0.85)',
                                        opacity: 0,
                                        transition: 'opacity 0.2s',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        justifyContent: 'center',
                                        padding: '20px',
                                        textAlign: 'center'
                                    }}>
                                        <h4 style={{ color: '#fff', marginBottom: '12px' }}>{item.title}</h4>
                                        <div style={{ fontSize: '0.85rem', color: '#ccc', marginBottom: '16px' }}>
                                            Shared by: <br />
                                            <span style={{ color: '#FF3D3D', fontWeight: 'bold' }}>
                                                {item.users.join(', ')}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* SECTION C: SHARED THEMES */}
                {sectionC && sectionC.length > 0 && (
                    <section className="results-section fade-in delay-3" style={{ marginTop: '80px', marginBottom: '100px' }}>
                        <div className="section-header" style={{ marginBottom: '32px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '16px' }}>
                            <h2 className="section-title" style={{ fontSize: '2rem', color: '#fff' }}>Shared Vibes</h2>
                            <p className="section-desc" style={{ color: '#888' }}>Themes your group connects on</p>
                        </div>

                        <div className="themes-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '32px' }}>
                            {sectionC.map((theme, idx) => (
                                <div key={idx} className="theme-card" style={{
                                    background: 'linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01))',
                                    padding: '32px',
                                    borderRadius: '24px',
                                    border: '1px solid rgba(255,255,255,0.05)'
                                }}>
                                    <h3 style={{ margin: '0 0 8px 0', fontSize: '1.4rem', color: '#ffb74d' }}>{theme.theme_name}</h3>
                                    <p style={{ margin: '0 0 24px 0', color: '#aaa', fontSize: '0.9rem', lineHeight: '1.6' }}>
                                        {theme.justification}
                                    </p>

                                    <div className="theme-movies" style={{ display: 'flex', gap: '12px' }}>
                                        {theme.recommended_movies.map(m => (
                                            <div key={m.movie_id} style={{ flex: 1 }}>
                                                {m.poster_url ? (
                                                    <img
                                                        src={m.poster_url}
                                                        alt={m.title}
                                                        style={{
                                                            width: '100%',
                                                            aspectRatio: '2/3', // Enforce poster ratio
                                                            height: 'auto',
                                                            objectFit: 'cover',
                                                            borderRadius: '8px',
                                                            marginBottom: '8px'
                                                        }}
                                                    />
                                                ) : (
                                                    <div style={{
                                                        background: '#222',
                                                        height: '140px',
                                                        borderRadius: '8px',
                                                        marginBottom: '8px',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        color: '#444',
                                                        fontSize: '0.8rem'
                                                    }}>
                                                        No Poster
                                                    </div>
                                                )}
                                                <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: '500' }}>{m.title}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                <div style={{ textAlign: 'center' }}>
                    <button className="cta-button primary" onClick={() => navigate('/create-group')}>
                        Start Over
                    </button>
                </div>
            </div>

            {/* Detail Modal */}
            {
                selectedMovie && (
                    <MovieDetailModal
                        movie={selectedMovie}
                        onClose={() => setSelectedMovie(null)}
                    />
                )
            }

            {/* Trailer Modal */}
            {
                trailerUrl && (
                    <div
                        style={{
                            position: 'fixed',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            background: 'rgba(0, 0, 0, 0.9)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            zIndex: 10000,
                            padding: '20px'
                        }}
                        onClick={() => setTrailerUrl(null)}
                    >
                        <div
                            style={{
                                position: 'relative',
                                width: '100%',
                                maxWidth: '1200px',
                                aspectRatio: '16/9',
                                background: '#000',
                                borderRadius: '12px',
                                overflow: 'hidden',
                                boxShadow: '0 20px 60px rgba(0,0,0,0.5)'
                            }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Close Button */}
                            <button
                                onClick={() => setTrailerUrl(null)}
                                style={{
                                    position: 'absolute',
                                    top: '16px',
                                    right: '16px',
                                    width: '40px',
                                    height: '40px',
                                    borderRadius: '50%',
                                    background: 'rgba(0,0,0,0.7)',
                                    border: '2px solid rgba(255,255,255,0.3)',
                                    color: 'white',
                                    fontSize: '1.5rem',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    zIndex: 10,
                                    transition: 'all 0.2s ease'
                                }}
                                onMouseEnter={(e) => {
                                    e.target.style.background = 'rgba(255,61,61,0.9)';
                                    e.target.style.transform = 'scale(1.1)';
                                }}
                                onMouseLeave={(e) => {
                                    e.target.style.background = 'rgba(0,0,0,0.7)';
                                    e.target.style.transform = 'scale(1)';
                                }}
                            >
                                ×
                            </button>

                            {/* YouTube Iframe */}
                            <iframe
                                width="100%"
                                height="100%"
                                src={trailerUrl.replace('watch?v=', 'embed/') + '?autoplay=1'}
                                title="Movie Trailer"
                                frameBorder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowFullScreen
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%'
                                }}
                            />
                        </div>
                    </div>
                )
            }
        </div >
    );
}

export default Results;
