import React, { useState, useEffect } from 'react';
import { ThemeProvider, CssBaseline, Container, Typography, Box, IconButton, AppBar, Toolbar } from '@mui/material';
import { lightTheme, darkTheme } from './theme';
import ImageUploader from './components/ImageUploader';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

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

    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
    const [isTablet, setIsTablet] = useState(window.innerWidth >= 768 && window.innerWidth < 1024);

    useEffect(() => {
      const handleResize = () => {
        setIsMobile(window.innerWidth < 768);
        setIsTablet(window.innerWidth >= 768 && window.innerWidth < 1024);
      };

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }, []);

    const containerWidth = isMobile ? '100%' : isTablet ? '70%' : '50%';


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
            <Container>
                <Box sx={{ my: 4 }}>
                    <ImageUploader containerWidth={containerWidth} />
                </Box>
            </Container>
        </ThemeProvider>
    );
};

export default App;
