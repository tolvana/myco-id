import React, { useEffect, useState } from 'react';
import styles from './Results.module.css';
import WikipediaIcon from '../icons/wikipedia';
import GbifIcon from '../icons/gbif';

// Props type definition
type ResultsProps = {
  results: [string, any][];
};

const Results: React.FC<ResultsProps> = ({ results }) => {
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
                    <span className={styles.scientificName}>{scientificName}</span>
                    <span>)</span>
                  </>
                ) : (
                  <span className={styles.scientificName}>{scientificName}</span>
                )}
              </div>

              {/* Aligning the icons to the right */}
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
          </div>
        );
      })}
    </div>
  );
};

export default Results;
