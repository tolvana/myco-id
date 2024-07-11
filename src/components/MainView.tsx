import React, {useState, useEffect, useMemo} from 'react';
import {ToastContainer, toast} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import {Box, Button, CircularProgress, Typography} from '@mui/material';
import Collapse from '@mui/material/Collapse';
import {useTheme} from '@mui/material/styles';

import Results from './Results';
import ImageClassifier from '../ImageClassifier';
import ImagePicker from './ImagePicker';

const MainView: React.FC = () => {

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
        setClassificationResults(null);

        try {
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

    const onResultClick = (result: [string, any]) => {
        console.log('Result clicked:', result);
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

                <form onSubmit={onSubmit}>

                    <ImagePicker onImages={(images) => {
                        setImageUrls(images)
                        setInvalidated(true)
                    }} />

                    <Box sx={{display: 'flex', justifyContent: 'center', mt: 3}}>
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            disabled={loading || downloading || !invalidated || imageUrls.every((url) => url === '')}
                            sx={{position: 'relative'}}
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
                    <Results results={results} onResultClick={onResultClick} />
                </Collapse>

                {downloading && (
                    <Box sx={{display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 3}}>
                        {downloadProgress !== null ? (
                            <CircularProgress variant="determinate" value={downloadProgress} />
                        ) : (
                            <CircularProgress />
                        )}
                        <Typography variant="body2" color="textSecondary" sx={{mt: 1}}>Downloading model...</Typography>
                    </Box>
                )}
        </>
    );
};

export default MainView;
