import './App.css';
import ImageUploader from './components/ImageUploader';
import React, { useState, useEffect } from 'react';

const App = () => {

  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [isTablet, setIsTablet] = useState(window.innerWidth >= 768 && window.innerWidth < 1024);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
      setIsTablet(window.innerWidth >= 768 && window.innerWidth < 1024);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const containerWidth = isMobile ? '100%' : isTablet ? '70%' : '50%';

  return (
    <div className="App">
      <header className="App-header">
        <p>Identify a mushroom</p>
        <ImageUploader containerWidth = {containerWidth} />
      </header>
    </div>
  );
}

export default App;
