import React, {useEffect} from 'react';
import {Box, IconButton} from '@mui/material';
import {styled} from '@mui/system';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import {useTheme} from '@mui/material/styles';


// example usage: <ImagesPreview urls={} onDelete={onDelete}/>
//
interface ImagesPreviewProps {
    // array of nullable strings
    urls: string[];
    onClick: (index: number) => void;
    onDelete: (index: number) => void;

};

const DeleteButton = styled(IconButton)({
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    '&:hover': {
        backgroundColor: 'rgba(255, 255, 255, 1)',
    },
});

const ImageContainer = styled('div')({
    position: 'relative',
    display: 'inline-block',
});

const ImagesPreview: React.FC<ImagesPreviewProps> = ({urls, onDelete, onClick}) => {

    const theme = useTheme();

    useEffect(() => {
        // check contents of each url
        const check = async () => {
        console.log(urls);
        for (let i = 0; i < urls.length; i++) {
            const img = await fetch(urls[i]);
            if (!img.ok || img.headers.get("Content-Type")?.startsWith("image") === false) {
                onDelete(i);
            }
        }
        };
        check();
    }, [urls[0]]);

    return (
        <Box
            sx={{
                display: 'grid',
                gridTemplateColumns: `repeat(4, 1fr)`,
                gap: 0.7, // Ensure equal spacing
            }}
        >
            {urls.map((url, index) => (
                <Box
                    key={index}
                    sx={{
                        position: 'relative',
                        borderRadius: 2,
                        overflow: 'hidden',
                    }}
                >
                    {url ? (
                        <ImageContainer>
                            <Box
                                component="img"
                                src={url}
                                alt={`Image ${index}`}
                                sx={{
                                    width: '100%',
                                    height: '100%',
                                    aspectRatio: '1/1',
                                    objectFit: 'cover'
                                }}
                            >
                            </Box>
                            <DeleteButton onClick={(_) => {onDelete(index)}} size="small">
                                <DeleteIcon />
                            </DeleteButton>
                        </ImageContainer>
                    ) : (

                        <Box
                            sx={{
                                width: '100%',
                                height: '100%',
                                aspectRatio: '1/1',
                                objectFit: 'cover',
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                            }}
                            bgcolor={theme.palette.action.disabledBackground}
                            onClick={() => {onClick(index)}}
                        >
                            <AddIcon
                                fontSize="large"
                                style={{
                                    width: '50%',
                                    height: '50%',
                                    color: theme.palette.action.disabled
                                }}
                            />
                        </Box>

                    )}
                </Box>
            ))}
        </Box>
    );


};

export default ImagesPreview;
