import React, {useEffect, useState} from 'react';
import {useParams, useNavigate} from 'react-router-dom';


const SpeciesDetailView: React.FC = () => {
    const {species} = useParams<Record<string, string>>();
    const navigate = useNavigate();
    console.log(species);

    const [speciesInfo, setSpeciesInfo] = useState<any | null>(null);

    useEffect(() => {
        fetch(`/info/${species}.json`)
            .then((response) => response.json())
            .then((data) => {
                console.log(data);
                setSpeciesInfo(data);
            }
        );
    }, [species]);

    const closeInfo = () => {
        navigate(-1); // Navigate back to the previous page
    };

    return (
        <div className="info-view">
            <div className="info-content">
                <button className="close-button" onClick={closeInfo}>
                    Close
                </button>
                <div className="info-details">
                {speciesInfo && (
                    <h2>{`${speciesInfo?.scientific_name}`}</h2>
                )}
                </div>
            </div>
        </div>
    );
};

export default SpeciesDetailView;
