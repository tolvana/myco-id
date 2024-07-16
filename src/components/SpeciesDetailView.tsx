import React, {useEffect, useState} from 'react';
import {useParams, useNavigate} from 'react-router-dom';
import {IconButton} from '@mui/material';
import WikipediaIcon from '../icons/wikipedia';
import GbifIcon from '../icons/gbif';


interface SpeciesDetailViewProps {
    setAppBarContent: React.Dispatch<React.SetStateAction<React.ReactNode>>;
}

const SpeciesDetailView: React.FC<SpeciesDetailViewProps> = ({setAppBarContent}) => {
    const {species} = useParams<Record<string, string>>();
    const navigate = useNavigate();
    console.log(species);

    const [speciesInfo, setSpeciesInfo] = useState<any | null>(null);

    useEffect(() => {
        if (!species) {
            return;
        }
        const name = species.replace('_', ' ');
        const capitalized = name.charAt(0).toUpperCase() + name.slice(1);
        setAppBarContent(capitalized);
    }, [setAppBarContent]);


    useEffect(() => {
        fetch(`/myco-id/info/${species}.json`)
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


    const wikiUrl = species ? `https://en.wikipedia.org/wiki/${encodeURIComponent(species)}` : "";

    const gbifId = speciesInfo?.gbif_id;
    const gbifUrl = gbifId ? `https://www.gbif.org/species/${encodeURIComponent(gbifId)}` : "";
    console.log(gbifUrl);

    return (
        <div className="info-view">
            <div className="info-content">
                <div className="info-details">
                    {speciesInfo && (

                        <>

                            <IconButton
                                edge="start"
                                color="inherit"
                                href={wikiUrl}
                                target="_blank"
                                rel="noreferrer"
                            >
                                <WikipediaIcon />
                            </IconButton>

                            <IconButton
                                edge="start"
                                color="inherit"
                                href={gbifUrl}
                                target="_blank"
                                rel="noreferrer"
                            >
                                <GbifIcon />
                            </IconButton>
                        </>


                    )}


                </div>
            </div>
        </div>
    );
};

export default SpeciesDetailView;
