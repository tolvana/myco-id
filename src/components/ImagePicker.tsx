import React, {useState, useRef} from 'react';
import {toast} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import {Box, Button} from '@mui/material';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';

import ImagesPreview from './ImagesPreview';

interface ImagePickerProps {
    imageUrls: string[];
    onImages: (urls: string[]) => void;
}

const ImagePicker: React.FC<ImagePickerProps> = ({imageUrls, onImages}) => {

    const [targetIdx, setTargetIdx] = useState<number | null>(null);
    const [lastCamera, setLastCamera] = useState<boolean>(true);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const cameraInputRef = useRef<HTMLInputElement>(null);

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
    }

    const onClick = (index: number) => {
        setTargetIdx(index);
        if (lastCamera) {
            cameraInputRef.current?.click();
        } else {
            fileInputRef.current?.click();
        }
    }

    return (
        <>
            <Box sx={{display: 'flex', justifyContent: 'center', mb: 3}}>

                <Button
                    variant="contained"
                    color="primary"
                    onClick={() => {
                        setLastCamera(false);
                        fileInputRef.current?.click()
                    }}
                    startIcon={<ImageSearchIcon />}
                    sx={{mr: 2}}
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
                style={{display: 'none'}}
                onChange={onFileChange}
                ref={cameraInputRef}
                capture="environment"
            />

            <input
                type="file"
                accept="image/*"
                style={{display: 'none'}}
                onChange={onFileChange}
                ref={fileInputRef}
                multiple={true}
            />

            <ImagesPreview urls={imageUrls} onDelete={onDelete} onClick={onClick} />

        </>
    );
};

export default ImagePicker;
