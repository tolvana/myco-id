import React, { useState, useRef } from 'react';
import axios from 'axios';
import Results from './Results';
import styles from './ImageUploader.module.css';
import { ToastContainer, toast } from 'react-toastify';  // Import toastify components
import 'react-toastify/dist/ReactToastify.css';  // Default styling


interface ImageUploaderProps {
    containerWidth: string;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ containerWidth }) => {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [classificationResults, setClassificationResults] = useState<Record<string, any> | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [imageUrl, setImageUrl] = useState<string>('');


    const fileInputRef = useRef<HTMLInputElement>(null);
    const cameraInputRef = useRef<HTMLInputElement>(null);

    const resetResults = () => {
        setClassificationResults(null);
    }

    const fetchImageFromUrl = async () => {
        if (!imageUrl) {
            return;
        }
        resetResults();
        setPreview(imageUrl);
    }


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
        let data = new FormData();
        if (file) {
            data.append('file', file);
        } else if (imageUrl)
            data.append('url', imageUrl);
        try {
            const response = await axios.post('http://192.168.1.98:5000/classify', data, {
                headers: {
                    'Content-Type': file ? 'multipart/form-data' : 'application/json',
                },
            });
            console.log(response.data);
            setClassificationResults(response.data);
        } catch (error) {
            console.error('Error:', error);
            // show snackbar / toast
            toast.error('Error uploading the image');  // Use toast to show error
        } finally {
            setLoading(false);
        }
    };

    const results = classificationResults ? Object.entries(classificationResults) : [];

    const containerStyle = { width: containerWidth, maxWidth: '768px'};
    console.log(containerStyle);

    return (
    <>
            <ToastContainer position="top-center" autoClose={4000} hideProgressBar={true}
                            newestOnTop={false} closeOnClick rtl={false} pauseOnFocusLoss
                            draggable pauseOnHover toastClassName={styles.toastCustom} />

        <div className="container mt-4" style={containerStyle}>
            <form onSubmit={onSubmit}>
                <div className="mb-3">

                    <div className={`${styles.buttonGroup} ${styles.centerDiv} mb-3`}>
                        <input
                            type="file"
                            className={`form-control ${styles.customFileInput}`}
                            id="file-upload"
                            onChange={onFileChange}
                            accept="image/*"
                            style={{ display: 'none' }}
                            ref={fileInputRef}
                        />
                        <button
                            type="button"
                            className={`btn btn-primary ${styles.buttonSpacing}`}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            Choose file
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => cameraInputRef.current?.click()}
                        >
                            Take picture
                        </button>
                    </div>
                    <div className={styles.centerDiv} style={{ marginTop: '30px' }}>
                        <input
                            type="text"
                            className={`form-control ${styles.urlInput}`}
                            placeholder="Enter image URL"
                            value={imageUrl}
                            onChange={(e) => setImageUrl(encodeURI(e.target.value))}
                        />
                    </div>
                    <div className={styles.centerDiv} style={{ marginTop: '10px' }}>
                        <button
                            type="button"
                            className="btn btn-info"
                            onClick={fetchImageFromUrl}
                        >
                            Load Image
                        </button>
                    </div>

                    {/* Placeholder for future functionality */}
                <input
                    type="file"
                    accept="image/*"
                    capture="environment"
                    style={{ display: 'none' }}
                    onChange={handleCameraInput}
                    ref={cameraInputRef}
                />
                </div>
                {preview && (
                    <div className="mb-3">
                        <img src={preview} alt="No image found." className={styles.roundedImage} />
                    </div>
                )}
                {file || imageUrl ? (
                    <button type="submit" className="btn btn-primary">Identify</button>
                ) : null}
            </form>
      {results.length > 0 ? (
        <Results results={results} />
      ) : ( loading ? (
                <div className={styles.spinner}>
                    <div className={styles.loader}></div>
                </div>
                ) : null
      )}
        </div>

    </>
    );
};

export default ImageUploader;
