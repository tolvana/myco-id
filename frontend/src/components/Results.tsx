import React, { useEffect, useState } from 'react';
import styles from './Results.module.css';
import WikipediaIcon from '../icons/wikipedia';
import GbifIcon from '../icons/gbif';

// Props type definition
type ResultsProps = {
  results: [string, any][];
};


// Component for showing gbif media of this species
const GbifObservations: React.FC<{ gbifId: string }> = ({ gbifId }) => {
    const [mediaResults, setMediaResults] = useState<any[]>([]);

    useEffect(() => {
        fetch(`https://api.gbif.org/v1/species/${gbifId}/media`)
            .then((response) => response.json())
            .then((data) => {
                setMediaResults(data.results);
            });
    }, [gbifId]);

    // Adjust number of images per row and maximum rows as needed
    const imagesPerRow = 4; // Number of images per row
    const maxRows = 3; // Maximum number of rows to display

    // Calculate number of columns dynamically based on images per row
    const gridTemplateColumns = `repeat(${imagesPerRow}, 1fr)`;

    return (
        <div className={styles.gbifObservations} style={{gridTemplateColumns}}>
            {mediaResults.slice(0, imagesPerRow * maxRows).map((result) => (
                <img
                    key={result.identifier}
                    src={result.identifier}
                    alt={result.title}
                    onClick={() => window.open(result.identifier)}
                    className={styles.thumbnail}
                />
            ))}
        </div>
    );
};

const Results: React.FC<ResultsProps> = ({ results }) => {
  console.log(results);
  return (
    <div className={styles.mainContainer}>
      {results.map(([scientificName, { probability, info }]) => {

        const normalizedName = scientificName.replace(" ", "_");
        const gbifId = info.gbif_id;
        const wikiUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(normalizedName)}`;
        const gbifUrl = `https://www.gbif.org/species/${encodeURIComponent(gbifId)}`;

        return (
          <div key={scientificName} className={styles.resultEntry}>
            <div className={styles.resultLabel}>
              <div className={styles.textContainer}>
                {info.common_names.eng ? (
                  <>
                    <span className={styles.vernacularName}>{info.common_names["eng"][0]}</span>
                    <span> (</span>
                    <span className={styles.scientificName}>{info.scientific_name}</span>
                    <span>)</span>
                  </>
                ) : (
                  <span className={styles.scientificName}>{info.scientific_name}</span>
                )}
              </div>

              <div className={styles.icons}>
                <a href={wikiUrl} target="_blank" rel="noopener noreferrer" className={styles.icon}>
                  <WikipediaIcon />
                </a>
                <a href={gbifUrl} target="_blank" rel="noopener noreferrer" className={styles.icon}>
                  <GbifIcon />
                </a>
              </div>
            </div>
            <div className={styles.progressBar}>
              <div className={styles.progressBarFill} style={{ width: `${probability * 100}%` }}>
                {`${(probability * 100).toFixed(1)}%`}
              </div>
            </div>
            <GbifObservations gbifId={gbifId} />
          </div>
        );
      })}
    </div>
  );
};

export default Results;
