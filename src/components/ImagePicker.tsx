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

interface ImagePickerProps {
    containerWidth: string;
}

const ImagePreview = styled('img')({
    width: '100%',
    maxHeight: '400px',
    objectFit: 'contain',
    borderRadius: '8px',
    marginTop: '20px',
});

const ImagePicker: React.FC<ImagePickerProps> = ({ containerWidth }) => {

    const [classificationResults, setClassificationResults] = useState<Record<string, any> | null>(null);
    const [imageUrls, setImageUrls] = useState<string[]>(['', '', '', '']);
    const [loading, setLoading] = useState<boolean>(false);
    const [downloadProgress, setDownloadProgress] = useState<number | null>(0);
    const [downloading, setDownloading] = useState<boolean>(false);
    const [lastCamera, setLastCamera] = useState<boolean>(true);
    const [targetIdx, setTargetIdx] = useState<number | null>(null);
    const [invalidated, setInvalidated] = useState<boolean>(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const cameraInputRef = useRef<HTMLInputElement>(null);

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

    const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        console.log(event.target.files);
        if (event.target.files) {

            if (event.target.files.length > 4) {
                toast.warn("You can only choose up to 4 images");
            } else if (event.target.files.length == 0) {
                return;
            }

            const newImageUrls = [...imageUrls];

            let j = 0;
            let localTargetIdx = targetIdx;

            for (let i = 0; i < Math.min(event.target.files.length, 4); i++) {
                const file = event.target.files[i];
                const url = URL.createObjectURL(file);
                const index = localTargetIdx ?? newImageUrls.findIndex((url) => url === '');
                localTargetIdx = null;
                setTargetIdx(null);
                if (index !== -1) {
                    newImageUrls[index] = url;
                } else {
                    newImageUrls[j++] = url;
                }
            }

            setImageUrls(newImageUrls);
            setInvalidated(true);
            setTargetIdx(null);
        }
    };

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

    const onDelete = (index: number) => {
        const newImageUrls = [...imageUrls];
        newImageUrls[index] = '';
        setImageUrls(newImageUrls);
        setInvalidated(true);
    }

    const onClick = (index: number) => {
        setTargetIdx(index);
        if (lastCamera) {
            cameraInputRef.current?.click();
        } else {
            fileInputRef.current?.click();
        }
    }

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
                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={() => {
                                setLastCamera(false);
                                fileInputRef.current?.click()
                            }}
                            startIcon={<ImageSearchIcon />}
                            sx={{ mr: 2 }}
                        >
                            Choose File
                        </Button>
                        <Button
                            variant="contained"
                            color="secondary"
                            onClick={() => {
                                setLastCamera(true);
                                cameraInputRef.current?.click()
                            }}
                            startIcon={<PhotoCameraIcon />}
                        >
                            Take Picture
                        </Button>
                    </Box>

                    <input
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={onFileChange}
                        ref={cameraInputRef}
                        capture="environment"
                    />

                    <input
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={onFileChange}
                        ref={fileInputRef}
                        multiple={true}
                    />

                    <ImagesPreview urls={imageUrls} onDelete={onDelete} onClick={onClick} />


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

export default ImagePicker;
