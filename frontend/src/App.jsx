import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import GroupCreation from './pages/GroupCreation';
import Results from './pages/Results';
import './index.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/create-group" element={<GroupCreation />} />
      <Route path="/results" element={<Results />} />
    </Routes>
  );
}

export default App;
