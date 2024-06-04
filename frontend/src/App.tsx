import React from 'react';
import './App.css';
import ImageUploader from './components/ImageUploader';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <p>Identify a mushroom</p>
        <ImageUploader />
      </header>
    </div>
  );
}

export default App;
