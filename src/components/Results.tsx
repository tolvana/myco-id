import React, {useEffect, useState, useRef} from 'react';
import {
    Grid,
    Typography,
    Box,
    IconButton,
    Skeleton,
    Link
} from '@mui/material';
import {styled} from '@mui/material/styles';
import Fade from '@mui/material/Fade';
import InfoIcon from '@mui/icons-material/Info';

import {Link as RouterLink} from 'react-router-dom';

// Props type definition
type ResultsProps = {
    results: [string, any][];
    onResultClick: (result: [string, any]) => void;
};

// Component for showing gbif media of this species
const GbifObservations: React.FC<{gbifId: string}> = ({gbifId}) => {
    const [mediaResults, setMediaResults] = useState<any[]>([]);
    const [imageLoaded, setImageLoaded] = useState<boolean[]>([]);
    const previousGbifIdRef = useRef<string | null>(null);

    useEffect(() => {
        if (previousGbifIdRef.current === gbifId) {
            return;
        }

        previousGbifIdRef.current = gbifId;

        fetch(`https://api.gbif.org/v1/species/${gbifId}/media`)
            .then((response) => response.json())
            .then((data) => {
                setMediaResults(data.results);
                setImageLoaded(new Array(data.results.length).fill(false));
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
            {
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
                        <Skeleton
                            variant="rectangular"
                            width="100%"
                            height="100%"
                            sx={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                paddingTop: '100%', // This keeps the aspect ratio 1:1
                                borderRadius: 1
                            }}
                        />

                        <Fade in={imageLoaded[index]}>
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
                                    '&:hover': {
                                        opacity: 0.8,
                                    },
                                }}
                            />
                        </Fade>
                    </Box>
                ))
            }
        </Box>
    );
};

const ResultsContainer = styled(Box)(({theme}) => ({
    marginTop: theme.spacing(4),
}));

const ResultEntry = styled(Box)(({theme}) => ({
    marginBottom: theme.spacing(2),
    padding: theme.spacing(2),
    borderRadius: "8px",
    backgroundColor: theme.palette.mode === "dark" ? "#222" : "#f6f6f6",
    boxShadow: theme.shadows[2],
}));

const Results: React.FC<ResultsProps> = ({results, onResultClick}) => {
    const resultsContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (results.length > 0 && resultsContainerRef.current) {
            resultsContainerRef.current.scrollIntoView({behavior: 'smooth'});
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
                                <Skeleton variant="text" width="40%" height={20} style={{marginTop: 8}} />
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
            {results.map(([scientificName, {probability, info}]) => {

                const gbifId = info.gbif_id;

                const normalizedName = scientificName.replace(' ', '_');
                // const wikiUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(normalizedName)}`;
                // const gbifUrl = `https://www.gbif.org/species/${encodeURIComponent(gbifId)}`;

                return (
                    <ResultEntry key={scientificName}>
                        <Grid container alignItems="center" spacing={2}>
                            <Grid item xs>
                                <Box>
                                    <Link variant="h6" color="inherit" component={RouterLink} to={`/details/${normalizedName}`}>
                                        {info.scientific_name}
                                    </Link>
                                    {info.common_names.eng && (
                                        <div>
                                            <Typography variant="body1" component="span">
                                                {info.common_names.eng[0]}
                                            </Typography>
                                        </div>
                                    )}
                                    <Typography variant="body2" sx={{marginTop: 1, marginBottom: 1}}>
                                        {`${(probability * 100).toFixed(1)}%`}
                                    </Typography>
                                </Box>
                            </Grid>
                            <Grid item>
                                <IconButton size="medium" onClick={() => onResultClick([scientificName, info])}>
                                    <InfoIcon />
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
