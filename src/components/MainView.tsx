import React, {useEffect, useMemo} from 'react';
import {ToastContainer, toast} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import {Box, Button, CircularProgress, Typography} from '@mui/material';
import Collapse from '@mui/material/Collapse';
import {useTheme} from '@mui/material/styles';

import Results from './Results';
import ImageClassifier from '../ImageClassifier';
import ImagePicker from './ImagePicker';

export interface MainViewState {
    classificationResults: Record<string, any> | null;
    imageUrls: string[];
    loading: boolean;
    downloadProgress: number | null;
    downloading: boolean;
    invalidated: boolean;
}

interface MainViewProps {
    state: Record<string, any>; // Replace Record<string, any> with the actual type of your state
    setState: React.Dispatch<React.SetStateAction<MainViewState>>;
}

const MainView: React.FC<MainViewProps> = ({state, setState}) => {

    const publicUrl = process.env.PUBLIC_URL;

    const sanePublicUrl = publicUrl.endsWith('/') ? publicUrl : `${publicUrl}/`;

    const modelUrl = `${sanePublicUrl}model.onnx`;
    const metadataUrl = `${sanePublicUrl}metadata.json`;

    const classifier = useMemo(() => new ImageClassifier(modelUrl, metadataUrl), []);

    const theme = useTheme();

    const onDownloadProgress = (progress: number) => {
        const newValue = progress === 100 ? null : progress;
        setState((prevState) => ({...prevState, downloading: true, downloadProgress: newValue}));
    };

    useEffect(() => {
        const loadModel = async () => {
            try {
                await classifier.load(onDownloadProgress);
            } catch (error) {
                console.error(error);
                toast.error("Unexpected error loading the model");
            } finally {
                setState((prevState) => ({...prevState, downloading: false}));
            }
        };

        loadModel();
    }, []);

    const onSubmit = async (event: React.FormEvent) => {
        event.preventDefault();

        setState((prevState) => ({...prevState, loading: true, classificationResults: null}));

        try {
            const images = state.imageUrls.filter((url: string) => url !== '').map((url: string) => {
                let img = new Image();
                img.src = url;
                return img;
            });

            const result = await classifier.classifyMultiple(images);
            setState((prevState) => ({...prevState, classificationResults: result}));
        } catch (error) {
            console.error('Error:', error);
            toast.error("Unexpected error processing the image(s)");
        } finally {
            setState((prevState) => ({...prevState, loading: false}));
        }
    };

    const onResultClick = (result: [string, any]) => {
        console.log('Result clicked:', result);
    };

    const results = state.classificationResults ? Object.entries(state.classificationResults) : [];

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

                <ImagePicker imageUrls={state.imageUrls} onImages={(images) => {
                    setState((prevState) => ({...prevState, imageUrls: images, invalidated: true}));
                }} />

                <Box sx={{display: 'flex', justifyContent: 'center', mt: 3}}>
                    <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        disabled={
                            state.loading
                            || state.downloading
                            || !state.invalidated
                            || state.imageUrls.every((url: string) => url === '')
                        }
                        sx={{position: 'relative'}}
                    >
                        {state.loading && (
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

            <Collapse in={results.length > 0 || state.loading}>
                <Results results={results} onResultClick={onResultClick} />
            </Collapse>

            {state.downloading && (
                <Box sx={{display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 3}}>
                    {state.downloadProgress !== null ? (
                        <CircularProgress variant="determinate" value={state.downloadProgress} />
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
