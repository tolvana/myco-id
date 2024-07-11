import React, { useState, useEffect } from 'react';
import { ThemeProvider, CssBaseline, Container, Typography, Box, IconButton, AppBar, Toolbar } from '@mui/material';
import { lightTheme, darkTheme } from './theme';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import MainView from './components/MainView';
import SpeciesDetailView from './components/SpeciesDetailView';

const App: React.FC = () => {
    const [darkMode, setDarkMode] = useState<boolean>(false);

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

    return (
        <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
            <CssBaseline />
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Identify a mushroom
                    </Typography>
                    <IconButton
                        edge="end"
                        color="inherit"
                        aria-label="mode"
                        onClick={handleThemeChange}
                    >
                        {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
                    </IconButton>
                </Toolbar>
            </AppBar>
            <Router>
            <Container maxWidth="sm">
                    <Box sx={{ my: 4 }}>
                        <Routes>
                            <Route path="/" element={<MainView />} />
                            <Route path="/details/:species" element={<SpeciesDetailView />} />
                        </Routes>
                    </Box>
                </Container>
            </Router>
        </ThemeProvider>
    );
};

export default App;
