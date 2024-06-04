import React, { useEffect, useState } from 'react';
import styles from './Results.module.css';

// Props type definition
type ResultsProps = {
  results: [string, any][];
};

// Declare the functional component with proper props handling
const Results: React.FC<ResultsProps> = ({ results }) => {

    // dict state for species data (keyed by scientific name)
    const [speciesData, setSpeciesData] = useState<Record<string, any>>({});

    // In your Results component
    return (
    <div className={styles.mainContainer}>
    {results.map(([scientificName, {probability, info}]) => (
      <div key={scientificName} className={styles.resultEntry}>
        <div className={styles.resultLabel}>
          {info.common_names.eng ? (
            <>
              <span className={styles.vernacularName}>{info.common_names["eng"][0]}</span>
              <span> (</span>
              <span className={styles.scientificName}>{scientificName}</span>
              <span>)</span>
            </>
          ) : (
            <span className={styles.scientificName}>{scientificName}</span>
          )}
        </div>
        <div className={styles.progressBar}>
          <div className={styles.progressBarFill} style={{ width: `${probability * 100}%` }}>
            {`${(probability * 100).toFixed(1)}%`}
          </div>
        </div>
      </div>
    ))}
    </div>
    );
};

export default Results;
