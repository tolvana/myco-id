import React, {useState, useEffect} from 'react';
import {ThemeProvider, CssBaseline, Container, Box, IconButton, AppBar, Toolbar} from '@mui/material';
import {lightTheme, darkTheme} from './theme';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import {Route, Routes, useLocation, useNavigate} from 'react-router-dom';
import Typography from '@mui/material/Typography';

import MainView, {MainViewState} from './components/MainView';
import SpeciesDetailView from './components/SpeciesDetailView';

const AppBarContent: React.FC<{
    handleBackClick: () => void;
    isDetailView: boolean;
    handleThemeChange: () => void;
    darkMode: boolean;
    children: React.ReactNode;
}> = ({ handleBackClick, isDetailView, handleThemeChange, darkMode, children }) => {
    return (
        <>
            {isDetailView && (
                <IconButton edge="start" color="inherit" aria-label="back" onClick={handleBackClick}>
                    <ArrowBackIcon />
                </IconButton>
            )}
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                {children}
            </Typography>
            <IconButton edge="end" color="inherit" aria-label="mode" onClick={handleThemeChange}>
                {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
        </>
    );
};

const App: React.FC = () => {
    const [darkMode, setDarkMode] = useState<boolean>(false);
    const [appBarContent, setAppBarContent] = useState<React.ReactNode>('Identify a mushroom');
    const location = useLocation();
    const navigate = useNavigate();

    const [mainState, setMainState] = useState<MainViewState>({
        classificationResults: null,
        imageUrls: ['', '', '', ''],
        loading: false,
        downloadProgress: 0,
        downloading: false,
        invalidated: false,
    });

    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        setDarkMode(mediaQuery.matches);

        const handleChange = (e: MediaQueryListEvent) => {
            setDarkMode(e.matches);
        };

        mediaQuery.addEventListener('change', handleChange);

        return () => {
            mediaQuery.removeEventListener('change', handleChange);
        };
    }, []);

    const handleThemeChange = () => {
        setDarkMode(!darkMode);
    };

    const handleBackClick = () => {
        navigate(-1);
    };

    const isDetailView = location.pathname.startsWith('/details/');

    return (
        <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
            <CssBaseline />
            <AppBar position="static">
                <Toolbar>
                    <AppBarContent
                        handleBackClick={handleBackClick}
                        isDetailView={isDetailView}
                        handleThemeChange={handleThemeChange}
                        darkMode={darkMode}
                    >
                        {appBarContent}
                    </AppBarContent>
                </Toolbar>
            </AppBar>
            <Container maxWidth="sm">
                <Box sx={{my: 4}}>
                    <Routes>
                        <Route path="/" element={<MainView state={mainState} setState={setMainState} setAppBarContent={setAppBarContent} />} />
                        <Route path="/details/:species" element={<SpeciesDetailView setAppBarContent={setAppBarContent} />} />
                    </Routes>
                </Box>
            </Container>
        </ThemeProvider>
    );
};

export default App;
