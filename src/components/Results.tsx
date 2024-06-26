import React, { useEffect, useState } from 'react';
import {
  Grid,
  Typography,
  Box,
  IconButton,
  CircularProgress,
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

  useEffect(() => {
    fetch(`https://api.gbif.org/v1/species/${gbifId}/media`)
      .then((response) => response.json())
      .then((data) => {
        setMediaResults(data.results);
        setLoading(false);
      });
  }, [gbifId]);

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
        mediaResults.slice(0, imagesPerRow * maxRows).map((result) => (
          <Box
            key={result.identifier}
            component="img"
            src={result.identifier}
            alt={result.title}
            onClick={() => window.open(result.identifier)}
            sx={{
              width: '100%', // Use 100% to fit within the grid cell
              height: 'auto', // Standard height
              aspectRatio: 1 / 1, // Square aspect ratio
              objectFit: 'cover', // Crop to square
              borderRadius: 1, // Rounded corners
              cursor: 'pointer',
              '&:hover': {
                opacity: 0.8,
              },
            }}
          />
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
  backgroundColor: theme.palette.mode == "dark" ? "#222" : "#f6f6f6",
  boxShadow: theme.shadows[2],
}));

const Results: React.FC<ResultsProps> = ({ results }) => {
  return (
    <ResultsContainer>
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
              <Grid item>
                <Typography variant="body2">{`${(probability * 100).toFixed(1)}%`}</Typography>
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
