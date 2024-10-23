import React, {useState, useRef, useEffect} from 'react';
import {toast} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import {Box, IconButton} from '@mui/material';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';
import DeleteIcon from '@mui/icons-material/Delete';

import ImagesPreview from './ImagesPreview';

interface ImagePickerProps {
    imageUrls: string[];
    onImages: (urls: string[]) => void;
}

const ImagePicker: React.FC<ImagePickerProps> = ({imageUrls, onImages}) => {

    const [targetIdx, setTargetIdx] = useState<number | null>(null);
    const [cameraAvailable, setCameraAvailable] = useState<boolean>(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Start camera when component mounts or when user clicks "Take Picture"
    const startCamera = async () => {
        const devices = await navigator.mediaDevices.enumerateDevices();
        setCameraAvailable(devices.some((device) => device.kind === 'videoinput'));
        console.log(devices);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: {exact: "environment"}, // Use back camera
                    width: {ideal: 1024},               // Adjust width (ideal is flexible)
                    height: {ideal: 1024},               // Adjust height
                },
            });
            if (videoRef.current) {
                console.log("streaming");
                videoRef.current.srcObject = stream;
            } else {
                console.log("no video ref");
            }
        } catch (error) {
            toast.error("Unable to access camera. Please allow camera permissions.");
        }
    };

    // Capture an image from the video feed
    const captureImage = () => {
        if (canvasRef.current && videoRef.current) {
            const videoWidth = videoRef.current.videoWidth;
            const videoHeight = videoRef.current.videoHeight;
            console.log("heh", videoWidth, videoHeight);

            // Set the canvas width and height to match the video dimensions
            canvasRef.current.width = videoWidth;
            canvasRef.current.height = videoHeight;

            const context = canvasRef.current.getContext('2d');
            if (context) {
                // Draw the video frame on the canvas
                console.log("capturing??");
                context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);

                // Capture the image from the canvas
                const url = canvasRef.current.toDataURL('image/png');
                console.log("captured", url);
                addImage(url);
            }
        }
    };

    // Add the captured image URL to the list of images
    const addImage = (url: string) => {
        const newImageUrls = [...imageUrls];
        const index = targetIdx ?? newImageUrls.findIndex((url) => url === '');
        if (index !== -1) {
            newImageUrls[index] = url;
        } else {
            newImageUrls.push(url);
        }
        onImages(newImageUrls);
        setTargetIdx(null);
    };

    const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
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

            onImages(newImageUrls);
            setTargetIdx(null);
        }
    };

    const onDelete = (index: number) => {
        const newImageUrls = [...imageUrls];
        newImageUrls[index] = '';
        onImages(newImageUrls);
    };

    const onClick = (index: number) => {
        setTargetIdx(index);
        fileInputRef.current?.click();
    }

    const removeAll = () => {
        onImages(['', '', '', '']);
    };

    useEffect(() => {

        startCamera();

        return () => {
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                const tracks = stream.getTracks();
                tracks.forEach((track) => track.stop());
            }
        };
    }, []);

    return (
        <>
            <input
                type="file"
                accept="image/*"
                style={{display: 'none'}}
                onChange={onFileChange}
                ref={fileInputRef}
                multiple={true}
            />

            {cameraAvailable && (

                <Box sx={{display: 'flex', justifyContent: 'center', mb: 3}}>
                    <video ref={videoRef} autoPlay playsInline style={{width: '100%', borderRadius: 10}} />
                </Box>

            )}

            <Box sx={{display: 'flex', justifyContent: 'space-between', mt: 2, width: '100%', paddingX: 4, paddingBottom: 2}}>

                {/* Remove Button */}
                <IconButton
                    onClick={removeAll}
                    sx={{
                        backgroundColor: 'secondary.main',
                        color: 'white',
                        width: '60px', // Fixed width
                        height: '60px', // Fixed height
                        '&:hover': {
                            backgroundColor: 'secondary.dark',
                        },
                    }}
                >
                    <DeleteIcon />
                </IconButton>

                {/* Capture Button */}
                <IconButton
                    onClick={captureImage}
                    sx={{
                        backgroundColor: 'primary.main',
                        color: 'white',
                        width: '80px', // Slightly wider for better visual balance
                        height: '80px', // Slightly taller for better visual balance
                        borderRadius: '50%',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)', // Add some shadow to the button
                        '&:hover': {
                            backgroundColor: 'primary.dark',
                        },
                    }}
                >
                    <PhotoCameraIcon sx={{fontSize: 32}} />
                </IconButton>

                {/* Add File Button */}
                <IconButton
                    onClick={() => fileInputRef.current?.click()}
                    sx={{
                        backgroundColor: 'primary.main',
                        color: 'white',
                        width: '60px', // Fixed width
                        height: '60px', // Fixed height
                        '&:hover': {
                            backgroundColor: 'primary.dark',
                        },
                    }}
                >
                    <ImageSearchIcon />
                </IconButton>
            </Box>

            {/* Hidden canvas to capture the image */}
            <canvas ref={canvasRef} style={{display: 'none'}} width={640} height={480}></canvas>

            <ImagesPreview urls={imageUrls} onDelete={onDelete} onClick={onClick} />
        </>
    );
};

export default ImagePicker;
