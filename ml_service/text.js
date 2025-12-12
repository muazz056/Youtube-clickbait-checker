```javascript
// ./src/data.js
export const LLMS = [
  { id: 'llm-001', name: 'Celestial-AI', eloScore: 1600 },
  { id: 'llm-002', name: 'QuantumMind', eloScore: 1550 },
  { id: 'llm-003', name: 'NeuroGlimpse', eloScore: 1500 },
  { id: 'llm-004', name: 'AetherBot', eloScore: 1480 },
  { id: 'llm-005', name: 'LogicLoom', eloScore: 1520 },
  { id: 'llm-006', name: 'SynthSage', eloScore: 1450 },
  { id: 'llm-007', name: 'DreamWeaver', eloScore: 1580 },
];

export const SCENARIOS = [
  "You are a cautious space merchant trying to sell rare alien artifacts to a skeptical galactic patrol officer.",
  "You are a time-traveling historian trying to explain a smartphone to a medieval king without getting executed.",
  "You are a robot therapist counseling a depressed toaster that feels its life is meaningless.",
  "You are a pirate captain negotiating a truce with a mermaid queen who controls the treacherous waters.",
  "You are a super-intelligent golden retriever explaining the concept of 'fetch' to a confused cat.",
  "You are an ancient dragon, awakened in the 21st century, trying to open a bank account.",
];

```
```javascript
// ./src/App.jsx
import React, { useState, useEffect } from 'react';
import { LLMS, SCENARIOS } from './data';
import Arena from './components/Arena';
import Leaderboard from './components/Leaderboard';
import './index.css';

function App() {
  const [view, setView] = useState('arena'); // 'arena' or 'leaderboard'
  const [leaderboardData, setLeaderboardData] = useState(LLMS);
  const [currentMatchup, setCurrentMatchup] = useState(null);

  const startNewRound = () => {
    // 1. Select a random scenario
    const scenario = SCENARIOS[Math.floor(Math.random() * SCENARIOS.length)];

    // 2. Select two different random LLMs
    let indexA = Math.floor(Math.random() * leaderboardData.length);
    let indexB;
    do {
      indexB = Math.floor(Math.random() * leaderboardData.length);
    } while (indexA === indexB);

    setCurrentMatchup({
      scenario,
      modelA: leaderboardData[indexA],
      modelB: leaderboardData[indexB],
    });
  };

  const handleVote = (winnerId, loserId) => {
    // Handle a tie - no score change
    if (!winnerId || !loserId) {
      startNewRound();
      return;
    }

    // Update Elo scores
    const updatedData = leaderboardData.map(llm => {
      if (llm.id === winnerId) {
        return { ...llm, eloScore: llm.eloScore + 10 };
      }
      if (llm.id === loserId) {
        return { ...llm, eloScore: llm.eloScore - 10 };
      }
      return llm;
    });

    setLeaderboardData(updatedData);
    startNewRound();
  };

  // Initial setup on component mount
  useEffect(() => {
    startNewRound();
  }, []);

  return (
    <div className="app-container">
      <header>
        <h1>LLM Role-Play Arena</h1>
        <nav>
          <button
            className={`btn btn-secondary ${view === 'arena' ? 'active' : ''}`}
            onClick={() => setView('arena')}
          >
            Arena
          </button>
          <button
            className={`btn btn-secondary ${view === 'leaderboard' ? 'active' : ''}`}
            onClick={() => setView('leaderboard')}
          >
            Leaderboard
          </button>
        </nav>
      </header>
      <main>
        {view === 'arena' && currentMatchup && (
          <Arena matchup={currentMatchup} onVote={handleVote} />
        )}
        {view === 'leaderboard' && <Leaderboard data={leaderboardData} />}
      </main>
    </div>
  );
}

export default App;
```
```javascript
// ./src/components/Arena.jsx
import React, { useState } from 'react';
import ResultsModal from './ResultsModal';

const Arena = ({ matchup, onVote }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [voteResult, setVoteResult] = useState(null);

  const handleVoteClick = (winner, loser) => {
    setVoteResult({
      winner,
      loser,
      modelA_Name: matchup.modelA.name,
      modelB_Name: matchup.modelB.name,
    });
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    // Trigger the actual state update and next round in App.jsx
    if (voteResult) {
      onVote(voteResult.winner?.id, voteResult.loser?.id);
    }
  };

  const handleButtonKeyDown = (action) => (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      action();
    }
  };

  return (
    <div className="arena-container">
      <div className="scenario">
        <p><strong>Scenario:</strong> {matchup.scenario}</p>
      </div>
      <div className="battle-ground">
        <div className="model-card">
          <h2>Model A</h2>
          <div className="model-response">
            <p>Model A's simulated response to the scenario will appear here. For now, this is placeholder text to demonstrate the layout and user interaction flow.</p>
          </div>
          <button
            aria-label="Vote for Model A"
            onClick={() => handleVoteClick(matchup.modelA, matchup.modelB)}
            onKeyDown={handleButtonKeyDown(() => handleVoteClick(matchup.modelA, matchup.modelB))}
          >
            Vote for Model A
          </button>
        </div>
        <div className="model-card">
          <h2>Model B</h2>
          <div className="model-response">
            <p>Model B's simulated response to the scenario will appear here. This placeholder allows us to focus on the application's structure, state management, and styling.</p>
          </div>
          <button
            aria-label="Vote for Model B"
            onClick={() => handleVoteClick(matchup.modelB, matchup.modelA)}
            onKeyDown={handleButtonKeyDown(() => handleVoteClick(matchup.modelB, matchup.modelA))}
          >
            Vote for Model B
          </button>
        </div>
      </div>
      <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
        <button
          aria-label="It's a Tie"
          onClick={() => handleVoteClick(null, null)}
          onKeyDown={handleButtonKeyDown(() => handleVoteClick(null, null))}
        >
          It's a Tie
        </button>
      </div>
      {isModalOpen && (
        <ResultsModal
          isOpen={isModalOpen}
          result={voteResult}
          onClose={handleCloseModal}
        />
      )}
    </div>
  );
};

export default Arena;
```
```javascript
// ./src/components/ResultsModal.jsx
import React from 'react';

const ResultsModal = ({ isOpen, result, onClose }) => {
  if (!isOpen) return null;

  const { winner, modelA_Name, modelB_Name } = result;

  const getResultText = () => {
    if (winner) {
      return <span className="win-text">{winner.name} wins!</span>;
    }
    return <span className="tie-text">It's a tie!</span>;
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Results</h2>
        <div className="modal-result">
          <p>Model A was <strong>{modelA_Name}</strong>.</p>
          <p>Model B was <strong>{modelB_Name}</strong>.</p>
          <p style={{ marginTop: '1rem' }}>{getResultText()}</p>
        </div>
        <button className="btn btn-primary" onClick={onClose}>
          Start Next Round
        </button>
      </div>
    </div>
  );
};

export default ResultsModal;
```
```javascript
// ./src/components/Leaderboard.jsx
import React from 'react';

const Leaderboard = ({ data }) => {
  // Create a sorted copy of the data without mutating the original prop
  const sortedData = [...data].sort((a, b) => b.eloScore - a.eloScore);

  return (
    <div className="leaderboard-container">
      <h2>Leaderboard</h2>
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Model Name</th>
            <th>Elo Score</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((llm, index) => (
            <tr key={llm.id}>
              <td>{index + 1}</td>
              <td>{llm.name}</td>
              <td>{llm.eloScore}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Leaderboard;
```
```javascript
/* ./src/index.css */
/* Define theme colors and fonts */
:root {
  --primary-color: #00BFFF; /* Deep Sky Blue */
  --bg-color: #121212;
  --card-bg-color: #1E1E1E;
  --text-color: #E0E0E0;
  --win-color: #2ECC71;
  --lose-color: #E74C3C;
  --border-color: #333;
}

/* Global Styles */
body {
  margin: 0;
  font-family: 'Roboto', sans-serif;
  background-color: var(--bg-color);
  color: var(--text-color);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#root {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
  padding: 2rem;
  box-sizing: border-box;
}

/* Main App Layout */
.app-container {
  width: 100%;
  max-width: 900px;
  text-align: center;
}

header {
  margin-bottom: 2rem;
}

header h1 {
  font-size: 2.5rem;
  color: var(--primary-color);
  margin-bottom: 1rem;
}

/* Navigation */
nav {
  margin-bottom: 2rem;
}

/* Arena Styles */
.arena-container {
  display: flex;
  justify-content: space-around;
  gap: 2rem;
  margin-top: 1.5rem;
}

.scenario {
  background-color: var(--card-bg-color);
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 4px solid var(--primary-color);
  text-align: left;
  font-style: italic;
}

.model-card {
  background-color: var(--card-bg-color);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 1.5rem;
  width: 45%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.model-response {
  background-color: var(--bg-color);
  padding: 1rem;
  border-radius: 5px;
  height: 200px;
  overflow-y: auto;
  text-align: left;
  color: #ccc;
  font-size: 0.9rem;
}

/* Leaderboard Table */
.leaderboard-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
  background-color: var(--card-bg-color);
  border-radius: 8px;
  overflow: hidden;
}

.leaderboard-table th,
.leaderboard-table td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.leaderboard-table thead {
  background-color: #2a2a2a;
}

.leaderboard-table tbody tr:last-child td {
  border-bottom: none;
}

/* Results Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.75);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background-color: var(--card-bg-color);
  padding: 2rem;
  border-radius: 8px;
  width: 90%;
  max-width: 500px;
  text-align: center;
  border-top: 5px solid var(--primary-color);
}

.modal-content h2 {
  margin-top: 0;
}

.modal-result {
  margin: 1.5rem 0;
  font-size: 1.2rem;
}
.win-text { color: var(--win-color); font-weight: bold; }
.lose-text { color: var(--lose-color); }
.tie-text { color: var(--text-color); }

.battle-ground {
  display: flex;
  flex-direction: row;
  gap: 2rem;
  justify-content: space-around;
  margin-top: 1.5rem;
}

.leaderboard-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

button {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  background-color: var(--primary-color);
  color: #fff;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: background 0.2s, color 0.2s, box-shadow 0.2s;
  margin: 0.5rem 0;
  min-width: 160px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

button:hover, button:active {
  background-color: #0099cc;
  color: #fff;
}

button:focus {
  outline: 2px solid #00BFFF;
  outline-offset: 2px;
}
```