import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MovieDetailModal from '../components/MovieDetailModal';

function Results() {
    const location = useLocation();
    const navigate = useNavigate();
    const { results } = location.state || {};
    const [selectedMovie, setSelectedMovie] = useState(null);
    const [trailerUrl, setTrailerUrl] = useState(null);
    const [infoMovie, setInfoMovie] = useState(null);

    if (!results) {
        return (
            <div className="cinefuse-app">
                <div className="content-wrapper" style={{ textAlign: 'center', paddingTop: '100px' }}>
                    <h1 className="scenario-title">No Results Found</h1>
                    <button className="cta-button primary" onClick={() => navigate('/')}>Go Home</button>
                </div>
            </div>
        );
    }

    const {
        section_a_top_recommendations: sectionA,
        section_b_common_watchlist: sectionB,
        section_c_shared_interests: sectionC,
        section_d_watchlist_inspired: sectionD,
        section_e_hybrid1_picks: sectionE,
        section_f_hybrid2_picks: sectionF
    } = results;

    const userMap = location.state?.userMap || {};

    // Info Icon Component - Shows only metadata
    const InfoIcon = ({ movie, style = {} }) => (
        <div
            onClick={(e) => { e.stopPropagation(); setInfoMovie(movie); }}
            style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'rgba(223, 1, 57, 0.15)',
                border: '1px solid rgba(223, 1, 57, 0.3)',
                color: 'rgba(223, 1, 57, 0.9)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.9rem',
                fontWeight: '600',
                cursor: 'pointer',
                opacity: 0,
                transition: 'opacity 0.2s, transform 0.2s',
                zIndex: 10,
                ...style
            }}
            className="info-icon"
            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >i</div>
    );

    return (
        <div className="cinefuse-app page-transition">
            <style>{`
                .movie-card-hover:hover .info-icon { opacity: 1 !important; }
                .hide-scrollbar::-webkit-scrollbar { display: none; }
                .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
                .movie-card-hover:hover .trailer-btn { opacity: 1 !important; }
            `}</style>

            <div className="results-container" style={{ padding: '0', maxWidth: '100%', margin: '0' }}>

                {/* TOP PICK HERO SECTION (Rank #1) - ORIGINAL DESIGN */}
                {sectionA && sectionA.length > 0 && (() => {
                    const topMovie = sectionA[0];
                    console.log('🎬 TOP MOVIE DATA:', topMovie);
                    console.log('📝 Overview:', topMovie.Overview || topMovie.overview);
                    return (
                        <div className="hero-recommendation-desktop" style={{
                            position: 'relative',
                            width: '100vw',
                            height: '85vh',
                            marginLeft: 'calc(-50vw + 50%)',
                            marginRight: 'calc(-50vw + 50%)',
                            overflow: 'hidden',
                            marginBottom: '60px',
                            boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
                        }}>
                            {/* Background Image - Backdrop Priority */}
                            <div style={{
                                position: 'absolute',
                                top: 0, left: 0, right: 0, bottom: 0,
                                backgroundImage: `url(${topMovie.backdrop_url || topMovie.poster_url})`,
                                backgroundSize: 'cover',
                                backgroundPosition: 'center top',
                                filter: 'brightness(0.5) saturate(1.1)'
                            }} />

                            {/* Gradient Overlay - Sadece Alta Yumuşak Geçiş */}
                            <div style={{
                                position: 'absolute',
                                top: 0, left: 0, right: 0, bottom: 0,
                                background: 'linear-gradient(to bottom, transparent 0%, transparent 60%, rgba(0,0,0,0.4) 85%, rgba(0,0,0,0.9) 100%), linear-gradient(to right, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 50%, transparent 70%)',
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
                                alignItems: 'flex-end',
                                textAlign: 'right',
                                zIndex: 2
                            }}>
                                {/* Rank Badge */}
                                <div className="rank-badge" style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '10px',
                                    background: 'rgba(223, 1, 57, 0.15)',
                                    border: '1px solid rgba(223, 1, 57, 0.3)',
                                    color: 'rgba(223, 1, 57, 0.9)',
                                    padding: '8px 20px',
                                    borderRadius: '50px',
                                    fontSize: '0.75rem',
                                    fontWeight: '500',
                                    letterSpacing: '0.2em',
                                    textTransform: 'uppercase',
                                    marginBottom: '24px',
                                    width: 'fit-content',
                                    backdropFilter: 'blur(5px)',
                                    fontFamily: 'var(--font-sans)'
                                }}>
                                    Top Pick
                                </div>

                                {/* Title & Info Icon */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px' }}>
                                    <h1 style={{
                                        fontSize: '4.5rem',
                                        fontWeight: '300',
                                        lineHeight: '1.1',
                                        color: '#fff',
                                        maxWidth: '1200px',
                                        textShadow: '0 2px 12px rgba(0,0,0,0.4)',
                                        margin: 0,
                                        fontFamily: 'var(--font-serif)',
                                        letterSpacing: '-0.02em'
                                    }}>
                                        {topMovie.title}
                                    </h1>
                                    <InfoIcon movie={topMovie} style={{ fontSize: '3rem' }} />
                                </div>

                                {/* Metadata */}
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '24px', marginBottom: '32px', fontSize: '1rem', color: '#b0b0b0', fontFamily: 'var(--font-sans)' }}>
                                    <span style={{
                                        color: '#e0e0e0',
                                        fontWeight: '400',
                                        fontSize: '1.1rem',
                                        letterSpacing: '-0.01em'
                                    }}>
                                        {Math.round((topMovie.group_score / 5) * 100)}% Match
                                    </span>
                                    {topMovie.genres && (
                                        <>
                                            <span style={{ opacity: 0.3 }}>|</span>
                                            <span style={{ fontWeight: '300', fontSize: '0.95rem' }}>{topMovie.genres.replace(/\|/g, ', ')}</span>
                                        </>
                                    )}
                                </div>

                                {/* Overview - No Title */}
                                <div style={{ marginBottom: '32px', maxWidth: '800px' }}>
                                    <p style={{
                                        fontSize: '1.1rem',
                                        lineHeight: '1.7',
                                        color: '#c0c0c0',
                                        textShadow: '0 1px 3px rgba(0,0,0,0.3)',
                                        fontFamily: 'var(--font-sans)',
                                        fontWeight: '300'
                                    }}>
                                        {topMovie.Overview || topMovie.overview || "No plot overview available."}
                                    </p>
                                </div>

                                {/* AI Explanation */}
                                {topMovie.group_explanation && (
                                    <div style={{ marginBottom: '48px', maxWidth: '800px', background: 'rgba(255, 61, 61, 0.08)', padding: '24px', borderRadius: '8px', borderLeft: '2px solid rgba(255, 61, 61, 0.3)' }}>
                                        <h3 style={{ fontSize: '0.7rem', color: '#FF3D3D', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '500' }}>Why This Film</h3>
                                        <p style={{ fontSize: '1rem', fontStyle: 'italic', color: '#b0b0b0', margin: 0, lineHeight: '1.8', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>
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
                                                background: '#C41E3A',
                                                color: 'white',
                                                border: 'none',
                                                padding: '16px 36px',
                                                borderRadius: '9999px',
                                                fontSize: '1rem',
                                                fontWeight: '600',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '10px',
                                                transition: 'all 0.3s ease',
                                                boxShadow: '0 10px 30px rgba(196, 30, 58, 0.4)',
                                                fontFamily: 'var(--font-sans)',
                                                letterSpacing: '0.02em'
                                            }}
                                            onMouseEnter={(e) => {
                                                e.target.style.transform = 'translateY(-2px)';
                                                e.target.style.background = '#D63447';
                                                e.target.style.boxShadow = '0 15px 40px rgba(196, 30, 58, 0.6)';
                                            }}
                                            onMouseLeave={(e) => {
                                                e.target.style.transform = 'translateY(0)';
                                                e.target.style.background = '#C41E3A';
                                                e.target.style.boxShadow = '0 10px 30px rgba(196, 30, 58, 0.4)';
                                            }}
                                        >
                                            Watch Trailer
                                        </button>
                                    )}
                                    <button
                                        onClick={() => setSelectedMovie(topMovie)}
                                        style={{
                                            background: 'rgba(20, 20, 20, 0.6)',
                                            color: 'white',
                                            border: '1px solid rgba(255, 255, 255, 0.1)',
                                            padding: '16px 36px',
                                            borderRadius: '9999px',
                                            fontSize: '1rem',
                                            fontWeight: '500',
                                            cursor: 'pointer',
                                            backdropFilter: 'blur(10px)',
                                            transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                                            fontFamily: 'var(--font-sans)',
                                            letterSpacing: '0.02em'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.target.style.background = 'rgba(30, 30, 30, 0.8)';
                                            e.target.style.borderColor = 'rgba(223, 1, 57, 0.3)';
                                            e.target.style.transform = 'translateY(-2px)';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.background = 'rgba(20, 20, 20, 0.6)';
                                            e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                                            e.target.style.transform = 'translateY(0)';
                                        }}
                                    >
                                        More Details
                                    </button>
                                </div>
                            </div>
                        </div>
                    );
                })()}

                {/* TOP 10 RECOMMENDATIONS - HORIZONTAL SCROLL */}
                {sectionA && sectionA.length > 1 && (
                    <section style={{ marginBottom: '80px', width: '100vw', marginLeft: 'calc(-50vw + 50%)' }}>
                        <div style={{ padding: '0 40px', marginBottom: '24px' }}>
                            <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '8px', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>Top 10 Recommendations</h2>
                            <p style={{ color: '#888', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>Additional films for your group</p>
                        </div>

                        <div className="hide-scrollbar" style={{ display: 'flex', overflowX: 'auto', gap: '24px', padding: '10px 40px', scrollBehavior: 'smooth' }}>
                            {sectionA.slice(1).map((movie, idx) => (
                                <div key={movie.movie_id || idx} className="movie-card-hover" style={{ flex: '0 0 280px', position: 'relative', borderRadius: '16px', overflow: 'hidden', aspectRatio: '2/3', background: '#1a1a1a', border: '1px solid rgba(255,255,255,0.05)', cursor: 'pointer', transition: 'transform 0.3s' }}
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-8px)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>

                                    <img src={movie.poster_url} alt={movie.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />

                                    <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: '50%', background: 'linear-gradient(to top, rgba(0,0,0,0.9), transparent)', pointerEvents: 'none' }} />

                                    <div style={{ position: 'absolute', top: '12px', left: '12px', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', color: '#fff', padding: '4px 10px', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 'bold' }}>#{idx + 2}</div>

                                    <InfoIcon movie={movie} />

                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '16px' }}>
                                        <h4 style={{ margin: '0 0 4px 0', fontSize: '1rem', lineHeight: '1.3' }}>{movie.title}</h4>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                                            <span style={{ color: '#ccc' }}>{movie.release_date?.split('-')[0]}</span>
                                            <span style={{ color: 'var(--primary-red)', fontWeight: '600' }}>{movie.group_score > 1 ? Math.round((movie.group_score / 5) * 100) : Math.round(movie.group_score * 100)}%</span>
                                        </div>
                                    </div>

                                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', gap: '12px', opacity: 0, transition: 'opacity 0.2s' }} className="trailer-btn">
                                        {movie.trailer_url && (
                                            <button onClick={(e) => { e.stopPropagation(); setTrailerUrl(movie.trailer_url); }} style={{ padding: '12px 24px', borderRadius: '20px', background: '#C41E3A', border: 'none', color: 'white', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem', boxShadow: '0 4px 12px rgba(196, 30, 58, 0.4)', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.background = '#D63447'} onMouseLeave={(e) => e.target.style.background = '#C41E3A'}>Trailer</button>
                                        )}
                                        <button onClick={(e) => { e.stopPropagation(); setSelectedMovie(movie); }} style={{ padding: '10px 20px', borderRadius: '20px', background: 'rgba(255,255,255,0.9)', border: 'none', color: '#000', fontWeight: 'bold', cursor: 'pointer' }}>View Details</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* WATCHLIST - HORIZONTAL SCROLL */}
                {sectionB && sectionB.length > 0 && (
                    <section style={{ marginTop: '80px', width: '100vw', marginLeft: 'calc(-50vw + 50%)' }}>
                        <div style={{ padding: '0 40px', marginBottom: '24px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '16px' }}>
                            <h2 style={{ fontSize: '2rem', color: '#fff', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>Shared Watchlists</h2>
                            <p style={{ color: '#888', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>Films you've already marked to watch</p>
                        </div>

                        <div className="hide-scrollbar" style={{ display: 'flex', overflowX: 'auto', gap: '24px', padding: '10px 40px', scrollBehavior: 'smooth' }}>
                            {sectionB.map((item) => (
                                <div key={item.movie_id} className="movie-card-hover" style={{ flex: '0 0 280px', position: 'relative', borderRadius: '16px', overflow: 'hidden', aspectRatio: '2/3', background: '#1a1a1a', border: '1px solid rgba(255,255,255,0.05)', cursor: 'pointer', transition: 'transform 0.3s' }}
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>

                                    <img src={item.poster_url} alt={item.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />

                                    <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: '50%', background: 'linear-gradient(to top, rgba(0,0,0,0.9), transparent)', pointerEvents: 'none' }} />

                                    <InfoIcon movie={item} />

                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '16px' }}>
                                        <h4 style={{ margin: '0 0 8px 0', fontSize: '1rem' }}>{item.title}</h4>
                                        <div style={{ fontSize: '0.75rem', color: '#ccc', marginBottom: '8px' }}>
                                            {item.release_date?.split('-')[0]} • <span style={{ color: 'var(--primary-red)', fontWeight: '600' }}>{item.users.join(', ')}</span>
                                        </div>
                                    </div>

                                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0, transition: 'opacity 0.2s' }} className="trailer-btn">
                                        {item.trailer_url && (
                                            <button onClick={(e) => { e.stopPropagation(); setTrailerUrl(item.trailer_url); }} style={{ padding: '12px 24px', borderRadius: '20px', background: '#C41E3A', border: 'none', color: 'white', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem', boxShadow: '0 4px 12px rgba(196, 30, 58, 0.4)', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.background = '#D63447'} onMouseLeave={(e) => e.target.style.background = '#C41E3A'}>Trailer</button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* SECTION D: WATCHLIST-INSPIRED RECOMMENDATIONS */}
                {sectionD && sectionD.length > 0 && (
                    <section style={{ marginTop: '80px', marginBottom: '100px', width: '100vw', marginLeft: 'calc(-50vw + 50%)' }}>
                        <div style={{ padding: '0 40px', marginBottom: '32px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '16px' }}>
                            <h2 style={{ fontSize: '2rem', color: '#fff', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>Based On Your Watchlists</h2>
                            <p style={{ color: '#888', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>New discoveries inspired by your saved films</p>
                        </div>

                        <div style={{ padding: '0 40px', overflowX: 'auto', display: 'flex', gap: '24px', paddingBottom: '20px' }}>
                            {sectionD.map((item, idx) => (
                                <div key={idx} style={{ flex: '0 0 240px', position: 'relative', borderRadius: '12px', overflow: 'hidden', aspectRatio: '2/3', cursor: 'pointer', transition: 'transform 0.3s' }}
                                    className="movie-card-hover"
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>

                                    <img src={item.poster_url} alt={item.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />

                                    <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: '40%', background: 'linear-gradient(to top, rgba(0,0,0,0.95), transparent)' }} />

                                    <InfoIcon movie={item} />

                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '16px' }}>
                                        <p style={{ margin: 0, fontSize: '1rem', fontWeight: '600', lineHeight: '1.3' }}>{item.title}</p>
                                        {item.group_score && (
                                            <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#C41E3A', fontWeight: 'bold' }}>
                                                {item.group_score > 1 ? Math.round((item.group_score / 5) * 100) : Math.round(item.group_score * 100)}% Match
                                            </p>
                                        )}
                                    </div>

                                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0, transition: 'opacity 0.2s' }} className="trailer-btn">
                                        {item.trailer_url && (
                                            <button onClick={(e) => { e.stopPropagation(); setTrailerUrl(item.trailer_url); }} style={{ padding: '12px 24px', borderRadius: '20px', background: '#C41E3A', border: 'none', color: 'white', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem', boxShadow: '0 4px 12px rgba(196, 30, 58, 0.4)', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.background = '#D63447'} onMouseLeave={(e) => e.target.style.background = '#C41E3A'}>Trailer</button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* SECTION E: HYBRID 1 PICKS */}
                {sectionE && sectionE.length > 0 && (
                    <section style={{ marginTop: '80px', marginBottom: '100px', width: '100vw', marginLeft: 'calc(-50vw + 50%)' }}>
                        <div style={{ padding: '0 40px', marginBottom: '32px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '16px' }}>
                            <h2 style={{ fontSize: '2rem', color: '#fff', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>Item Similarity Picks</h2>
                            <p style={{ color: '#888', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>Based on similar movies you've enjoyed (IBCF + Content)</p>
                        </div>

                        <div style={{ padding: '0 40px', overflowX: 'auto', display: 'flex', gap: '24px', paddingBottom: '20px' }}>
                            {sectionE.map((item, idx) => (
                                <div key={idx} style={{ flex: '0 0 240px', position: 'relative', borderRadius: '12px', overflow: 'hidden', aspectRatio: '2/3', cursor: 'pointer', transition: 'transform 0.3s' }}
                                    className="movie-card-hover"
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>

                                    <img src={item.poster_url} alt={item.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                    <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: '40%', background: 'linear-gradient(to top, rgba(0,0,0,0.95), transparent)' }} />
                                    <InfoIcon movie={item} />

                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '16px' }}>
                                        <p style={{ margin: 0, fontSize: '1rem', fontWeight: '600', lineHeight: '1.3' }}>{item.title}</p>
                                        {item.group_score && (
                                            <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#C41E3A', fontWeight: 'bold' }}>
                                                {item.group_score > 1 ? Math.round((item.group_score / 5) * 100) : Math.round(item.group_score * 100)}% Match
                                            </p>
                                        )}
                                    </div>

                                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0, transition: 'opacity 0.2s' }} className="trailer-btn">
                                        {item.trailer_url && (
                                            <button onClick={(e) => { e.stopPropagation(); setTrailerUrl(item.trailer_url); }} style={{ padding: '12px 24px', borderRadius: '20px', background: '#C41E3A', border: 'none', color: 'white', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem', boxShadow: '0 4px 12px rgba(196, 30, 58, 0.4)', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.background = '#D63447'} onMouseLeave={(e) => e.target.style.background = '#C41E3A'}>Trailer</button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* SECTION F: HYBRID 2 PICKS */}
                {sectionF && sectionF.length > 0 && (
                    <section style={{ marginTop: '80px', marginBottom: '100px', width: '100vw', marginLeft: 'calc(-50vw + 50%)' }}>
                        <div style={{ padding: '0 40px', marginBottom: '32px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '16px' }}>
                            <h2 style={{ fontSize: '2rem', color: '#fff', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>User Taste Picks</h2>
                            <p style={{ color: '#888', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>Based on users with similar preferences (UBCF + Content)</p>
                        </div>

                        <div style={{ padding: '0 40px', overflowX: 'auto', display: 'flex', gap: '24px', paddingBottom: '20px' }}>
                            {sectionF.map((item, idx) => (
                                <div key={idx} style={{ flex: '0 0 220px', position: 'relative', borderRadius: '12px', overflow: 'hidden', aspectRatio: '2/3', cursor: 'pointer', transition: 'transform 0.3s' }}
                                    className="movie-card-hover"
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>

                                    <img src={item.poster_url} alt={item.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                    <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: '40%', background: 'linear-gradient(to top, rgba(0,0,0,0.95), transparent)' }} />
                                    <InfoIcon movie={item} />

                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '16px' }}>
                                        <p style={{ margin: 0, fontSize: '1rem', fontWeight: '600', lineHeight: '1.3' }}>{item.title}</p>
                                        {item.group_score && (
                                            <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#C41E3A', fontWeight: 'bold' }}>
                                                {item.group_score > 1 ? Math.round((item.group_score / 5) * 100) : Math.round(item.group_score * 100)}% Match
                                            </p>
                                        )}
                                    </div>

                                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0, transition: 'opacity 0.2s' }} className="trailer-btn">
                                        {item.trailer_url && (
                                            <button onClick={(e) => { e.stopPropagation(); setTrailerUrl(item.trailer_url); }} style={{ padding: '12px 24px', borderRadius: '20px', background: '#C41E3A', border: 'none', color: 'white', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem', boxShadow: '0 4px 12px rgba(196, 30, 58, 0.4)', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.background = '#D63447'} onMouseLeave={(e) => e.target.style.background = '#C41E3A'}>Trailer</button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* SHARED VIBES (Moved to end) */}
                {sectionC && sectionC.length > 0 && (
                    <section style={{ marginTop: '80px', marginBottom: '100px', width: '100vw', marginLeft: 'calc(-50vw + 50%)' }}>
                        <div style={{ padding: '0 40px', marginBottom: '32px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '16px' }}>
                            <h2 style={{ fontSize: '2rem', color: '#fff', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>Shared Interests</h2>
                            <p style={{ color: '#888', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>Themes your group connects on</p>
                        </div>

                        <div style={{ padding: '0 40px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '40px' }}>
                            {sectionC.map((theme, idx) => (
                                <div key={idx} style={{ background: 'rgba(20, 20, 20, 0.6)', padding: '32px', borderRadius: '16px', border: '1px solid rgba(223, 1, 57, 0.2)' }}>
                                    <h3 style={{ margin: '0 0 12px 0', fontSize: '1.4rem', color: 'rgba(223, 1, 57, 0.9)', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>
                                        {theme.theme_name.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ')}
                                    </h3>
                                    <p style={{ margin: '0 0 24px 0', color: '#d0d0d0', fontSize: '0.95rem', lineHeight: '1.6', fontFamily: 'var(--font-sans)', fontWeight: '300' }}>{theme.justification}</p>

                                    <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
                                        {theme.recommended_movies.map(m => (
                                            <div key={m.movie_id} className="movie-card-hover" style={{ width: '280px', position: 'relative', borderRadius: '12px', overflow: 'hidden', aspectRatio: '2/3', cursor: 'pointer', transition: 'transform 0.3s' }}
                                                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>

                                                <img src={m.poster_url} alt={m.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />

                                                <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: '40%', background: 'linear-gradient(to top, rgba(0,0,0,0.95), transparent)' }} />

                                                <InfoIcon movie={m} />

                                                <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '16px' }}>
                                                    <p style={{ margin: 0, fontSize: '1rem', fontWeight: '600', lineHeight: '1.3' }}>{m.title}</p>
                                                </div>

                                                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0, transition: 'opacity 0.2s' }} className="trailer-btn">
                                                    {m.trailer_url && (
                                                        <button onClick={(e) => { e.stopPropagation(); setTrailerUrl(m.trailer_url); }} style={{ padding: '12px 24px', borderRadius: '20px', background: '#C41E3A', border: 'none', color: 'white', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem', boxShadow: '0 4px 12px rgba(196, 30, 58, 0.4)', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.background = '#D63447'} onMouseLeave={(e) => e.target.style.background = '#C41E3A'}>Trailer</button>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <button className="cta-button primary" onClick={() => navigate('/create-group')}>Start Over</button>
                </div>
            </div>

            {/* METADATA INFO MODAL (!) */}
            {infoMovie && (
                <div onClick={() => setInfoMovie(null)} style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000, padding: '20px' }}>
                    <div onClick={(e) => e.stopPropagation()} style={{ background: 'linear-gradient(135deg, #1a1a1a, #2a2a2a)', borderRadius: '24px', padding: '40px', maxWidth: '1100px', width: '100%', maxHeight: '90vh', overflowY: 'auto', border: '1px solid rgba(223, 1, 57, 0.2)', boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)', position: 'relative' }}>
                        <button onClick={() => setInfoMovie(null)} style={{ position: 'absolute', top: '20px', right: '20px', background: 'rgba(223, 1, 57, 0.15)', border: '1px solid rgba(223, 1, 57, 0.3)', color: 'rgba(223, 1, 57, 0.9)', fontSize: '24px', width: '40px', height: '40px', borderRadius: '50%', cursor: 'pointer', zIndex: 1 }}>×</button>

                        <h2 style={{ margin: '0 0 32px 0', fontSize: '2rem', color: '#fff', paddingRight: '50px', fontFamily: 'var(--font-serif)', fontWeight: '400', letterSpacing: '-0.01em' }}>{infoMovie.title}</h2>

                        {/* FLEX LAYOUT: Poster + Info */}
                        <div style={{ display: 'flex', gap: '40px', alignItems: 'flex-start' }}>
                            {/* LEFT: Poster */}
                            {infoMovie.poster_url && (
                                <div style={{ flex: '0 0 300px' }}>
                                    <img src={infoMovie.poster_url} alt={infoMovie.title} style={{ width: '100%', borderRadius: '12px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)', marginBottom: '16px' }} />

                                    {/* Watch Trailer Button */}
                                    {infoMovie.trailer_url && (
                                        <button
                                            onClick={(e) => { e.stopPropagation(); setTrailerUrl(infoMovie.trailer_url); setInfoMovie(null); }}
                                            style={{
                                                width: '100%',
                                                padding: '14px 20px',
                                                background: '#C41E3A',
                                                border: 'none',
                                                borderRadius: '9999px',
                                                color: 'white',
                                                fontSize: '1rem',
                                                fontWeight: '600',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                gap: '8px',
                                                transition: 'all 0.3s ease',
                                                boxShadow: '0 4px 15px rgba(196, 30, 58, 0.4)',
                                                fontFamily: 'var(--font-sans)',
                                                letterSpacing: '0.02em'
                                            }}
                                            onMouseEnter={(e) => {
                                                e.target.style.transform = 'translateY(-2px)';
                                                e.target.style.background = '#D63447';
                                                e.target.style.boxShadow = '0 6px 20px rgba(196, 30, 58, 0.6)';
                                            }}
                                            onMouseLeave={(e) => {
                                                e.target.style.transform = 'translateY(0)';
                                                e.target.style.background = '#C41E3A';
                                                e.target.style.boxShadow = '0 4px 15px rgba(196, 30, 58, 0.4)';
                                            }}
                                        >
                                            Watch Trailer
                                        </button>
                                    )}
                                </div>
                            )}

                            {/* RIGHT: Info */}
                            <div style={{ flex: 1 }}>
                                <div style={{ marginBottom: '24px' }}>
                                    <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>Genres</h3>
                                    <p style={{ color: '#c0c0c0', margin: 0, fontFamily: 'var(--font-sans)', fontWeight: '300', lineHeight: '1.6' }}>{infoMovie.genres?.replace(/\|/g, ', ') || 'N/A'}</p>
                                </div>

                                {(infoMovie.Overview || infoMovie.overview) && (
                                    <div style={{ marginBottom: '24px' }}>
                                        <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>Overview</h3>
                                        <p style={{ color: '#c0c0c0', lineHeight: '1.7', margin: 0, fontFamily: 'var(--font-sans)', fontWeight: '300' }}>{infoMovie.Overview || infoMovie.overview}</p>
                                    </div>
                                )}

                                {(infoMovie.Director || infoMovie.director) && (
                                    <div style={{ marginBottom: '24px' }}>
                                        <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>Director</h3>
                                        <p style={{ color: '#c0c0c0', margin: 0, fontFamily: 'var(--font-sans)', fontWeight: '300' }}>{infoMovie.Director || infoMovie.director}</p>
                                    </div>
                                )}

                                {(infoMovie.Actors || infoMovie.actors || infoMovie.cast) && (
                                    <div style={{ marginBottom: '24px' }}>
                                        <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>Cast</h3>
                                        <p style={{ color: '#c0c0c0', margin: 0, fontFamily: 'var(--font-sans)', fontWeight: '300' }}>{infoMovie.Actors || infoMovie.actors || infoMovie.cast}</p>
                                    </div>
                                )}

                                {(infoMovie.Production_Countries || infoMovie.production_countries || infoMovie.country) && (
                                    <div style={{ marginBottom: '24px' }}>
                                        <h3 style={{ fontSize: '0.7rem', color: '#999', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontFamily: 'var(--font-sans)', fontWeight: '400' }}>Country</h3>
                                        <p style={{ color: '#c0c0c0', margin: 0, fontFamily: 'var(--font-sans)', fontWeight: '300' }}>{infoMovie.Production_Countries || infoMovie.production_countries || infoMovie.country}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {selectedMovie && (
                <MovieDetailModal
                    movie={selectedMovie}
                    userMap={userMap}
                    onClose={() => setSelectedMovie(null)}
                />
            )}

            {/* TRAILER MODAL */}
            {trailerUrl && (
                <div onClick={() => setTrailerUrl(null)} style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000, padding: '20px' }}>
                    <div onClick={(e) => e.stopPropagation()} style={{ position: 'relative', width: '100%', maxWidth: '1200px', aspectRatio: '16/9', background: '#000', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 20px 60px rgba(0,0,0,0.5)' }}>
                        <button onClick={() => setTrailerUrl(null)} style={{ position: 'absolute', top: '16px', right: '16px', width: '40px', height: '40px', borderRadius: '50%', background: 'rgba(0,0,0,0.7)', border: '2px solid rgba(255,255,255,0.3)', color: 'white', fontSize: '1.5rem', cursor: 'pointer', zIndex: 10 }}>×</button>
                        <iframe width="100%" height="100%" src={trailerUrl.replace('watch?v=', 'embed/') + '?autoplay=1'} title="Movie Trailer" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen style={{ position: 'absolute', top: 0, left: 0 }} />
                    </div>
                </div>
            )}
        </div>
    );
}

export default Results;
