import React, { useState, useRef, useEffect, useMemo } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Box, Button, CircularProgress, Container, styled, Typography } from '@mui/material';
import Collapse from '@mui/material/Collapse';
import { useTheme } from '@mui/material/styles';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';

import Results from './Results';
import ImageClassifier from '../ImageClassifier';
import ImagesPreview from './ImagesPreview';
import ImagePicker from './ImagePicker';

interface MainViewProps {
    containerWidth: string;
}

const ImagePreview = styled('img')({
    width: '100%',
    maxHeight: '400px',
    objectFit: 'contain',
    borderRadius: '8px',
    marginTop: '20px',
});

const MainView: React.FC<MainViewProps> = ({ containerWidth }) => {

    const [classificationResults, setClassificationResults] = useState<Record<string, any> | null>(null);
    const [imageUrls, setImageUrls] = useState<string[]>(['', '', '', '']);
    const [loading, setLoading] = useState<boolean>(false);
    const [downloadProgress, setDownloadProgress] = useState<number | null>(0);
    const [downloading, setDownloading] = useState<boolean>(false);
    const [invalidated, setInvalidated] = useState<boolean>(false);

    const classifier = useMemo(() => new ImageClassifier("model.onnx", "metadata.json"), []);

    const theme = useTheme();

    const onDownloadProgress = (progress: number) => {
        setDownloading(true);
        if (progress === 100) {
            setDownloadProgress(null);
        } else {
            setDownloadProgress(progress);
        }
    };

    useEffect(() => {
        const loadModel = async () => {
            try {
                await classifier.load(onDownloadProgress);
            } catch (error) {
                console.error(error);
                toast.error("Unexpected error loading the model");
            } finally {
                setDownloading(false);
            }
        };

        loadModel();
    }, []);

    const onSubmit = async (event: React.FormEvent) => {
        event.preventDefault();

        setLoading(true);

        try {
            const validImageUrls = imageUrls.filter((url) => url !== '');

            const images = imageUrls.filter((url) => url !== '').map((url) => {
                let img = new Image();
                img.src = url;
                return img;
            });

            const result = await classifier.classifyMultiple(images);
            setClassificationResults(result);
            setInvalidated(false);
        } catch (error) {
            console.error('Error:', error);
            toast.error("Unexpected error processing the image(s)");
        } finally {
            setLoading(false);
        }
    };

    const results = classificationResults ? Object.entries(classificationResults) : [];

    return (
        <>
            <ToastContainer
                position="top-center"
                autoClose={4000}
                hideProgressBar={true}
                newestOnTop={false}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme={theme.palette.mode}
            />

            <Container maxWidth="sm" sx={{ width: containerWidth }}>
                <form onSubmit={onSubmit}>

                    <ImagePicker onImages={(images) => {
                        setImageUrls(images)
                        setInvalidated(true)
                    }} />

                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            disabled={loading || downloading || !invalidated || imageUrls.every((url) => url === '')}
                            sx={{ position: 'relative' }}
                        >

                        {loading && (
                            <CircularProgress
                                size={24}
                                sx={{
                                    color: 'primary.contrastText',
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    marginTop: '-12px',
                                    marginLeft: '-12px',
                                }}
                            />
                        )}

                        Identify

                        </Button>
                    </Box>

                </form>

                <Collapse in={results.length > 0 || loading}>
                    <Results results={results} />
                </Collapse>

                {downloading && (
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 3 }}>
                    {downloadProgress !== null ? (
                        <CircularProgress variant="determinate" value={downloadProgress} />
                         ) : (

                        <CircularProgress/>
                         )}
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>Downloading model...</Typography>
                    </Box>
                )}
            </Container>
        </>
    );
};

export default MainView;
