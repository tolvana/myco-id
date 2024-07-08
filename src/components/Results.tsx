import React, { useEffect, useState, useRef } from 'react';
import {
  Grid,
  Typography,
  Box,
  IconButton,
  CircularProgress,
  Skeleton,
} from '@mui/material';
import WikipediaIcon from '../icons/wikipedia'; // Assuming these are custom icons
import GbifIcon from '../icons/gbif'; // Assuming these are custom icons
import { styled } from '@mui/material/styles';

// Props type definition
type ResultsProps = {
  results: [string, any][];
};

// Component for showing gbif media of this species
const GbifObservations: React.FC<{ gbifId: string }> = ({ gbifId }) => {
  const [mediaResults, setMediaResults] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [imageLoaded, setImageLoaded] = useState<boolean[]>([]);
  const previousGbifIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (previousGbifIdRef.current === gbifId) {
      return;
    }

    previousGbifIdRef.current = gbifId;
    setLoading(true);

    fetch(`https://api.gbif.org/v1/species/${gbifId}/media`)
      .then((response) => response.json())
      .then((data) => {
        setMediaResults(data.results);
        setImageLoaded(new Array(data.results.length).fill(false));
        setLoading(false);
      });
  }, [gbifId]);

  const handleImageLoad = (index: number) => {
    setImageLoaded((prev) => {
      const newLoaded = [...prev];
      newLoaded[index] = true;
      return newLoaded;
    });
  };

  const imagesPerRow = 4; // Number of images per row
  const maxRows = 1; // Maximum number of rows to display

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: `repeat(${imagesPerRow}, 1fr)`,
        gap: 1, // Ensure equal spacing
      }}
    >
      {loading ? (
        <CircularProgress />
      ) : (
        mediaResults.slice(0, imagesPerRow * maxRows).map((result, index) => (
          <Box
            key={result.identifier}
            sx={{
              position: 'relative', // Make sure the box is relatively positioned for absolute positioning of Skeleton
              width: '100%',
              paddingTop: '100%', // This makes a square box
              borderRadius: 1,
              overflow: 'hidden', // Ensure border radius applies to both image and skeleton
            }}
          >
            {!imageLoaded[index] && (
              <Skeleton
                variant="rectangular"
                width="100%"
                height="100%"
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  borderRadius: 1
                }}
              />
            )}
            <Box
              component="img"
              src={result.identifier}
              alt={result.title}
              onLoad={() => handleImageLoad(index)}
              onClick={() => window.open(result.identifier)}
              sx={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                position: 'absolute',
                top: 0,
                left: 0,
                borderRadius: 1,
                cursor: 'pointer',
                display: imageLoaded[index] ? 'block' : 'none',
                '&:hover': {
                  opacity: 0.8,
                },
              }}
            />
          </Box>
        ))
      )}
    </Box>
  );
};

const ResultsContainer = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(4),
}));

const ResultEntry = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(2),
  padding: theme.spacing(2),
  borderRadius: "8px",
  backgroundColor: theme.palette.mode === "dark" ? "#222" : "#f6f6f6",
  boxShadow: theme.shadows[2],
}));

const Results: React.FC<ResultsProps> = ({ results }) => {
  const resultsContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    console.log(results);
    if (results.length > 0 && resultsContainerRef.current) {
      resultsContainerRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [JSON.stringify(results)]);

  if (results.length === 0) {
    return (
      <ResultsContainer ref={resultsContainerRef}>
        <ResultEntry>
          <Grid container alignItems="center" spacing={2}>
            <Grid item xs>
              <Box>
                <Skeleton variant="text" width="60%" height={30} />
                <Skeleton variant="text" width="40%" height={20} />
                <Skeleton variant="text" width="40%" height={20} style={{ marginTop: 8 }} />
              </Box>
            </Grid>
          </Grid>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: `repeat(4, 1fr)`,
              gap: 1, // Ensure equal spacing
              marginTop: 2,
            }}
          >
            {[...Array(4)].map((_, index) => (
              <Skeleton
                key={index}
                variant="rectangular"
                width="100%"
                height={0}
                sx={{
                  paddingTop: '100%', // This keeps the aspect ratio 1:1
                  borderRadius: 1
                }}
              />
            ))}
          </Box>
        </ResultEntry>
      </ResultsContainer>
    );
  }

  return (
    <ResultsContainer ref={resultsContainerRef}>
      {results.map(([scientificName, { probability, info }]) => {
        const normalizedName = scientificName.replace(' ', '_');
        const gbifId = info.gbif_id;
        const wikiUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(normalizedName)}`;
        const gbifUrl = `https://www.gbif.org/species/${encodeURIComponent(gbifId)}`;

        return (
          <ResultEntry key={scientificName}>
            <Grid container alignItems="center" spacing={2}>
              <Grid item xs>
                <Box>
                  {info.common_names.eng ? (
                    <>
                      <div>
                        <Typography variant="h6" component="span">
                          {info.common_names['eng'][0]}
                        </Typography>
                      </div>

                      <div>
                        <Typography variant="body2" component="span">
                          {' '}
                          {info.scientific_name}
                        </Typography>
                      </div>
                    </>
                  ) : (
                    <Typography variant="h6" component="span">
                      {info.scientific_name}
                    </Typography>
                  )}
                  <Typography variant="body2" sx={{ marginTop: 1, marginBottom: 1 }}>
                    {`${(probability * 100).toFixed(1)}%`}
                  </Typography>
                </Box>
              </Grid>
              <Grid item>
                <IconButton
                  component="a"
                  href={wikiUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <WikipediaIcon />
                </IconButton>
                <IconButton
                  component="a"
                  href={gbifUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <GbifIcon />
                </IconButton>
              </Grid>
            </Grid>
            <GbifObservations gbifId={gbifId} />
          </ResultEntry>
        );
      })}
    </ResultsContainer>
  );
};

export default Results;
