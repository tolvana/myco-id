import React, { useState, useRef, useEffect, useMemo } from 'react';
import Results from './Results';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import InferenceService from '../InferenceService';
import { Box, Button, CircularProgress, Container, TextField, styled, Typography } from '@mui/material';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';

interface ImageUploaderProps {
    containerWidth: string;
}

const ImagePreview = styled('img')({
    width: '100%',
    maxHeight: '400px',
    objectFit: 'contain',
    borderRadius: '8px',
    marginTop: '20px',
});

const ImageUploader: React.FC<ImageUploaderProps> = ({ containerWidth }) => {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [classificationResults, setClassificationResults] = useState<Record<string, any> | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [imageUrl, setImageUrl] = useState<string>('');
    const [downloadProgress, setDownloadProgress] = useState<number | null>(0);
    const [downloading, setDownloading] = useState<boolean>(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const cameraInputRef = useRef<HTMLInputElement>(null);

    const inferenceService = useMemo(() => new InferenceService("model.onnx", "metadata.json"), []);

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
                await inferenceService.loadModel(onDownloadProgress);
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

    const fetchImageFromUrl = async () => {
        if (!imageUrl) {
            return;
        }
        resetResults();
        setPreview(imageUrl);
    };

    const handleCameraInput = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            setFile(file);  // Save the file in the state
            const reader = new FileReader();
            reader.onload = function(e) {
                resetResults();
                setPreview(e.target!.result as string); // Set preview to display the image
            };
            reader.readAsDataURL(file);
        }
    };

    const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            const selectedFile = event.target.files[0];
            setFile(selectedFile);
            resetResults();
            setPreview(URL.createObjectURL(selectedFile));
        } else {
            setPreview(null);
        }
    };

    const onSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        if (!file && !imageUrl) {
            return;
        }

        if (loading || classificationResults) {
            return;
        }

        setLoading(true);

        try {
            let image = new Image();
            image.src = preview!;

            image.onload = async () => {
                const result = await inferenceService.runInference(image);
                setClassificationResults(result);
                setLoading(false);
            };
        } catch (error) {
            console.error('Error:', error);
            toast.error('Error processing the image');
        }
    };

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
                        onChange={handleCameraInput}
                        ref={cameraInputRef}
                    />

                    {preview && (
                        <ImagePreview src={preview} alt="No image found." />
                    )}
                    {(file || imageUrl) && (
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

export default ImageUploader;
