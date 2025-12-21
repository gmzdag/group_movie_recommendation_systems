import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import GroupCreation from './pages/GroupCreation';
import './index.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/create-group" element={<GroupCreation />} />
    </Routes>
  );
}

export default App;
