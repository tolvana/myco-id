import React, { useState, useRef, useEffect, useMemo } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Box, Button, CircularProgress, Container, styled, Typography } from '@mui/material';
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
    const [preview, setPreview] = useState<string | null>(null);
    const [classificationResults, setClassificationResults] = useState<Record<string, any> | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [imageUrl, setImageUrl] = useState<string>('');
    const [downloadProgress, setDownloadProgress] = useState<number | null>(0);
    const [downloading, setDownloading] = useState<boolean>(false);

    const [imageUrls, setImageUrls] = useState<string[]>(['', '', '', '']);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const cameraInputRef = useRef<HTMLInputElement>(null);

    const classifier = useMemo(() => new ImageClassifier("model.onnx", "metadata.json"), []);

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
                toast.error((error as Error).message);
            } finally {
                setDownloading(false);
            }
        };

        loadModel();
    }, []);

    const resetResults = () => {
        setClassificationResults(null);
    };

    const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            const selectedFile = event.target.files[0];
            resetResults();
            const url = URL.createObjectURL(selectedFile);
            console.log(url);
            setImageUrl(url);
            const index = imageUrls.findIndex((url) => url === '');
            if (index !== -1) {
                const newImageUrls = [...imageUrls];
                newImageUrls[index] = url;
                setImageUrls(newImageUrls);
            } else {
                const newImageUrls = [...imageUrls];
                newImageUrls[0] = url;
                setImageUrls(newImageUrls);
            }
        }
    };

    const onSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        // if all images are empty, return
        if (imageUrls.every((url) => url === '')) {
            return;
        }

        if (loading || classificationResults) {
            return;
        }

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
        } catch (error) {
            console.error('Error:', error);
            toast.error('Error processing the image');
        } finally {
            setLoading(false);
        }
    };

    const onDelete = (index: number) => {
        const newImageUrls = [...imageUrls];
        newImageUrls[index] = '';
        setImageUrls(newImageUrls);
    }

    const results = classificationResults ? Object.entries(classificationResults) : [];

    return (
        <>
            <ToastContainer position="top-center" autoClose={4000} hideProgressBar={true}
                            newestOnTop={false} closeOnClick rtl={false} pauseOnFocusLoss
                            draggable pauseOnHover />

            <Container maxWidth="sm" sx={{ width: containerWidth }}>
                <form onSubmit={onSubmit}>
                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                        <input
                            type="file"
                            id="file-upload"
                            onChange={onFileChange}
                            accept="image/*"
                            style={{ display: 'none' }}
                            ref={fileInputRef}
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={() => fileInputRef.current?.click()}
                            startIcon={<ImageSearchIcon />}
                            sx={{ mr: 2 }}
                        >
                            Choose File
                        </Button>
                        <Button
                            variant="contained"
                            color="secondary"
                            onClick={() => cameraInputRef.current?.click()}
                            startIcon={<PhotoCameraIcon />}
                        >
                            Take Picture
                        </Button>
                    </Box>

                    <input
                        type="file"
                        accept="image/*"
                        capture="environment"
                        style={{ display: 'none' }}
                        onChange={onFileChange}
                        ref={cameraInputRef}
                    />

                    <ImagesPreview urls={imageUrls} onDelete={onDelete} />

                    {imageUrl && (
                        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                            <Button
                                type="submit"
                                variant="contained"
                                color="primary"
                                disabled={loading}
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
                    )}
                </form>
                {(results.length > 0 || loading) && (
                    <Results results={results} />
                )}
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
